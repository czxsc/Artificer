# Artificer — Engineering Case Study

> **What it is:** an art‑recognition system. Point a camera at a painting and it tells you what you're looking at — the artist, genre, and style — and (in the planned end state) writes you a short, grounded blurb about the work.
>
> **Why I built it this way:** the interesting problem isn't "can a model label a painting." It's *what should the system do when it isn't sure* — when the exact painting is in the database, when it's a lookalike that isn't, or when it's something the model has never seen. That question drove almost every engineering decision below.
>
> **Status at time of writing:** the machine‑learning core is built and measured (Phases 0–1). The retrieval‑augmented generation layer and the full‑stack app (Phases 2–3) are scaffolded and designed but not yet implemented. This document is honest about that line.

---

## The Product Idea in One Picture

The target behavior is a small decision tree, not a single classifier:

```
        photo of an artwork
                │
   ┌────────────┴─────────────┐
   │  Is this exact piece in   │
   │      our database?        │
   └────────────┬─────────────┘
       yes      │      no
        │       │
 return the     │
 real metadata  │
        ┌───────┴────────┐
        │ Confident about │
        │ style / artist? │
        └───────┬────────┘
          yes   │   no
           │    │
    predict     │
    labels      │
                └──► say "I'm not sure"
                     (out‑of‑distribution)
```

Everything in Phases 0 and 1 exists to answer two of those branches well: **"is the exact piece here?"** (retrieval) and **"what are its labels?"** (classification) — and to figure out which technique is better at each.

---

## Table of Contents

- [Phase 0 — Getting the data honest](#phase-0--getting-the-data-honest)
- [Phase 0.5 — Building evaluation sets that reflect reality](#phase-05--building-evaluation-sets-that-reflect-reality)
- [Phase 1 — Two competing pipelines: Retrieval vs Classification](#phase-1--two-competing-pipelines-retrieval-vs-classification)
- [Results so far](#results-so-far)
- [Cross-cutting engineering problems](#cross-cutting-engineering-problems)
- [Phase 2 — RAG (planned)](#phase-2--rag-planned)
- [Phase 3 — Full-stack app (planned)](#phase-3--full-stack-app-planned)
- [What I'd tell an interviewer I learned](#what-id-tell-an-interviewer-i-learned)

---

## Phase 0 — Getting the data honest

**Goal:** pick a dataset, understand exactly what's in it, and clean it *before* trusting a single training number.

I used the [WikiArt dataset on Hugging Face](https://huggingface.co/datasets/huggan/wikiart) — about **81,444 paintings**, each labeled with an artist, a genre, and a style.

### The problem I found before training anything

Instead of jumping into a model, I first wrote a label audit (`scripts/dataset_analysis.py`) that tallies every label, ranks it, and flags imbalance. Two things fell out immediately:

| Label type | # of labels | Largest class | Its share |
|---|---|---|---|
| Artist | 129 | **"Unknown Artist"** | **51.5%** |
| Genre | 11 | "Unknown Genre" | 20.2% |
| Style | 27 | Impressionism | 16.0% |

**More than half of the entire dataset had no known artist.** If I'd trained naively, the model's easiest way to get a high artist score would have been to shout "Unknown Artist" at everything — a number that looks fine on paper and is useless in the product.

### The decision

Drop every "Unknown Artist" row for the classification work. That takes the dataset from 81,444 → **39,530 paintings** where the artist label actually means something. The audit script keeps *both* views (`full_dataset` and `known_artist_only`) so the removal is documented and reversible, not silent. The distribution is still long‑tailed after cleaning (the top‑10 artists are ~29% of the data; the rarest artist, Canaletto, has 42 works) — so imbalance stays a first‑class concern downstream, but it's no longer dominated by a meaningless bucket.

### The tricky part: splitting a long‑tailed dataset fairly

An 80/10/10 train/val/test split sounds trivial until you want it **stratified** — every artist, genre, and style represented in roughly the same proportion across all three splits. The catch: many combinations are *rare*. A painting that's the only example of its exact (artist, genre, style) triple can't be split three ways.

My fix (`data/split_utils.py`) is a **cascading stratification key**. For each painting I try increasingly coarse buckets until one is common enough to split safely:

```
artist ∣ genre ∣ style   →   style ∣ artist   →   style   →   artist   →   genre   →   "__rare__"
   (most specific)                                                              (catch-all)
```

Each row lands in the finest bucket that has at least `min_combo_count` (default 5) members, then splits happen within buckets. There's also a guard that shrinks the test/val counts for tiny groups so **every group keeps at least one training example** — you never accidentally move a rare artist entirely into the test set where the model has never seen them.

**Artifacts produced:** per‑label distribution CSVs, a human‑readable `bias_report.md`, a machine‑readable `summary.json`, and the reproducible train/val/test manifests (seeded, so the split is identical on every machine).

---

## Phase 0.5 — Building evaluation sets that reflect reality

**Goal:** measure the system the way it'll actually be used — on phone photos in a museum, not clean scans.

A model that scores 95% on pristine dataset images can fall apart on a real photo that's slightly blurry, tilted, cropped by the frame, or color‑shifted by gallery lighting. So I built a corruption library (`data/image_corruptions.py`) with **10 "museum photo" recipes**:

- Gaussian blur (mild + strong)
- Center crops (10% and 25% — simulating a frame cutting the edges)
- Warm color shift + desaturation (gallery lighting)
- Tilt left/right (±8°)
- Perspective shear left/right (photographing at an angle)

I then defined **two different test sets on purpose**, because they answer different questions:

| Test set | What it is | Question it answers |
|---|---|---|
| `known_artworks_test` | Held‑out paintings from classes the model *did* train on | **Generalization** — can it label a painting it's never seen? |
| `exact_artworks` | A random sample of paintings the model *did* train on | **Memorization / upper bound** — "we've catalogued this exact piece; can retrieval find it?" |

Each one also gets a **degraded** twin (all 10 corruptions applied).

### A storage decision worth calling out

The full degraded set is 10× the images. Materializing every corrupted JPEG to disk would be gigabytes of redundant files. Because the corruptions are **deterministic**, I don't store them — `eval_retrieval.py` applies them on the fly at evaluation time, and the build script only renders a **handful of preview images** for eyeballing that the recipes look right (`--materialize preview`). Full rendering is available behind a flag for when it's genuinely needed. Small decision, but it's the difference between a repo you can clone and one you can't.

---

## Phase 1 — Two competing pipelines: Retrieval vs Classification

**Goal:** don't assume which architecture is right — build one of each, hold everything else constant, and let the numbers decide.

Both paths start from the same place (a frozen vision model turning an image into an embedding) and diverge on what they do with it.

### Path A — Retrieval ("find the nearest known painting")

Embed the query image, then do a **cosine nearest‑neighbor search** against a database of known artworks. The top matches' metadata becomes the answer; for labels, the neighbors *vote*.

Key design choices (`ml/models/retrieval/`):

- **Encoder‑agnostic engine.** A `BaseImageEncoder` abstract class defines the one method an encoder must implement (image → normalized vector). The index, search, and voting logic are shared. Swapping CLIP for DINOv2, SigLIP, or ResNet‑50 is a one‑line registry change — which is exactly what you want when the whole point is a fair model bake‑off.
- **Brute‑force cosine search, on purpose.** At WikiArt scale (~40k items) an exact dot‑product scan is milliseconds and returns the *true* nearest neighbors. Reaching for an approximate index (FAISS/HNSW) here would add a dependency and an accuracy caveat to buy speed I don't need yet. It's a deliberate "right tool for the current scale" call, noted in the code so it's clearly a choice and not an oversight.
- **Two voting strategies:** flat (every neighbor equal) and cosine‑weighted (closer matches count more), so aggregation itself is a variable I can test.

### Path B — Classification ("predict the labels directly")

Freeze CLIP, attach three small trainable heads — one each for artist, genre, and style — and train them to predict labels (`ml/models/clip/classifier.py`, `scripts/train_clip_classifier.py`).

Key design choices:

- **Frozen backbone + lightweight heads.** I don't fine‑tune the 150M‑parameter CLIP encoder; I train three tiny MLPs on top of it. Faster, far less prone to overfitting on an imbalanced 40k set, and it keeps Path A and Path B comparable because they share the *same* frozen embedding.
- **The optimization that made iteration fast:** because the encoder is frozen, an image's embedding *never changes between epochs*. So I encode every image **once**, cache the vectors to disk (keyed by an MD5 of the model name + exact row set), and every training epoch trains only the tiny heads on cached vectors. Epochs drop from minutes of GPU encoding to **seconds**. The expensive step runs one time, ever, per split.
- **Multi‑task learning.** One model, three heads, loss = the average of three cross‑entropies. Artist, genre, and style share signal, so learning them together is cheaper and mutually reinforcing.
- **Per‑head early stopping — the subtle one.** While training I noticed the three heads learn at *different speeds*: genre and style plateau early, but the artist head (129 classes, the hardest) keeps improving. A single "stop when the average stops improving" rule would cut off the artist head too soon. So I (a) snapshot **each head at its own best‑validation epoch** and reassemble the final model from those, and (b) watch the **artist head specifically** for the early‑stopping decision. This is a real accuracy win that a naive training loop leaves on the table.
- **Crash‑safe metrics.** Loss/F1 curves and JSON metrics are written to disk **after every epoch**, so an interrupted or early‑stopped run never loses its history.

### Keeping the comparison fair

The CLIP *retrieval* encoder deliberately reproduces the exact projection the CLIP *classifier* uses (vision pooler → visual projection → L2 normalize), so both paths operate on the identical 512‑dimensional embedding. Any difference in results is due to **retrieval vs classification**, not two different flavors of CLIP.

---

## Results so far

### Path B — Classification (held‑out test set, artist‑Unknown removed)

| Head | Top‑1 accuracy | Macro F1 | Top‑5 accuracy |
|---|---|---|---|
| **Genre** | 79.7% | 0.78 | 99.1% |
| **Style** | 71.1% | 0.71 | 97.9% |
| **Artist** (129 classes) | 71.3% | 0.66 | **91.1%** |

Trained 16 epochs then early‑stopped. Train‑vs‑validation F1 gap stayed under 0.10 the whole way — **no meaningful overfitting**, which is the payoff of freezing the backbone and only training small heads. The top‑5 numbers matter for the product: even when the model's single best artist guess is wrong, the right answer is in its top 5 **91% of the time** — plenty good for a "did you mean…?" UI.

### Path A — Retrieval (label vote over top‑10 neighbors, clean queries)

| Encoder | Artist | Genre | Style | Mean | Encode latency |
|---|---|---|---|---|---|
| **CLIP** | 61.8% | 79.3% | 72.9% | **71.3%** | 15.4 ms |
| **DINOv2** | 51.6% | 76.3% | 63.3% | 63.8% | 29.1 ms |

Search itself is ~0.2 ms/query — the brute‑force call was the right choice. CLIP beats DINOv2 on *semantic* labeling, which is expected: CLIP was trained to align images with text, so its embedding space is organized around meaning; DINOv2 is self‑supervised and shines more at pure visual/instance matching (which is what the exact‑retrieval and degraded tests, still to be run, are designed to probe).

### The head‑to‑head that matters

On the **same** held‑out set, comparing "predict the artist directly" vs "let the 10 nearest neighbors vote":

| Label | Classification (Path B) | Retrieval (Path A, CLIP) |
|---|---|---|
| Artist | **71.3%** | 61.8% |
| Genre | 79.7% | 79.3% (tie) |
| Style | 71.1% | 72.9% |

**Early read:** a trained classification head clearly wins on the hard, fine‑grained task (artist), while retrieval holds its own on the coarser ones. That's a genuinely useful finding for the final architecture — it suggests the two paths are complementary rather than either‑or, which loops right back to the decision tree at the top.

---

## Cross-cutting engineering problems

Things that weren't in the plan but ate real time — the kind of stuff that never shows up in a tidy architecture diagram:

- **A Windows‑only crash from library import order.** On this machine, importing `datasets` (which loads PyArrow's native libraries) *after* the CUDA build of PyTorch triggers a hard DLL load‑order access violation — the process just dies. The fix is to import `datasets` *first* in every entry‑point script. It's now a documented, load‑bearing comment at the top of each script so a future me doesn't "tidy up the imports" and reintroduce the crash.
- **Not re‑downloading 30GB of data.** `data/wikiart_utils.py` looks for the dataset in the local Hugging Face cache first — it globs for the Arrow shards, stitches them back together, and only falls back to a network download if nothing is cached. Reproducible on a fresh machine, fast on a warm one.
- **Reproducibility throughout.** Fixed seeds, split manifests checked into `outputs/`, and metrics/config written alongside every result, so any number in this document can be regenerated from the manifests.

---

## Phase 2 — RAG (planned)

*Scaffolded, not yet built.* The plan: take the labels the model produces, **rewrite them into a structured knowledge query** (`ml/rag/rewriter.py`), retrieve grounded facts about the artwork/artist (`ml/rag/retriever.py`), and have an LLM write the final blurb (`ml/rag/pipeline.py`). Grounding the generation in retrieved facts is the guard against an LLM confidently inventing art history. Considering swapping in richer metadata (e.g. the Met's open collection) at this stage.

---

## Phase 3 — Full-stack app (planned)

*Designed, currently stubs.* A FastAPI backend (`/scan` for image → matches, plus `collections`, `daily_game`, and an `engineering` endpoint), a database of Artworks/Artists/Collections, and a React front end with a live camera. The `daily_game` and `collections` endpoints hint at the consumer framing I'm aiming for — a daily "guess the artwork" game and a personal collection, on top of the scan core. Production concerns already on the checklist: upload size limits, error/loading states, and API logging.

---

## What I'd tell an interviewer I learned

- **Audit the data before you trust any metric.** The single most important decision in this project — dropping the 51% "Unknown Artist" rows — happened *before* a model was trained, and it's the reason the accuracy numbers mean anything.
- **Design the evaluation to match reality.** Blur, tilt, crop, and color‑shift test sets, plus the deliberate split between "generalization" and "memorization" questions, are what turn a demo into a measurable system.
- **Let architecture be an experiment, not an assumption.** Building retrieval *and* classification behind a fair, shared embedding surfaced that they're complementary — a conclusion I couldn't have reached by picking one upfront.
- **The small optimizations compound.** Caching frozen embeddings (minutes → seconds per epoch), per‑head early stopping (free accuracy), on‑the‑fly corruptions (a cloneable repo), and crash‑safe metrics each sound minor; together they're the difference between a project I could iterate on quickly and one I couldn't.
```
