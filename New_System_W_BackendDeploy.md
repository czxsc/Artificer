# Artificer — Build Plan (MVP)

**One-liner:** Point your camera at a painting; Artificer identifies it — or its style/artist/period — and generates a short, grounded "curator's note."

**Portfolio goal:** Demonstrate end-to-end software engineering — a containerized ML service with a vector database, a real API, a live frontend, and a reproducible deploy — not model training. The modeling is done; this plan is about shipping it.

---

## Status

**Done**
- **Dataset (Phase 0):** WikiArt (~81K works). Label tally + distribution/bias analysis, cleaning, stratified 80/10/10 split.
- **Core model:** frozen CLIP ViT-B/32 backbone; multi-task MLP heads (style / genre / artist); pgvector top-5 retrieval; four-tier confidence cascade.
- **Results:** style@1 0.721, genre@1 0.792, artist@1 0.728, artist@5 0.919. → These + the design rationale go on the **About page**.
  <br>*(Source of truth: `outputs/phase1/clip_classifier_metrics.json`, from the 2026-07-31 retrain on the rebuilt workstation — same splits, verified by manifest hash. `CASE_STUDY.md` still quotes the original run, which is ~1pp different and whose checkpoint no longer exists.)*

**Left to build:** package the model as a service, add RAG, build a minimal frontend, deploy it live.

**Descoped → Future work:** gallery, minigame, alternate-backbone experiments (DINO/ResNet/ViT), robustness benchmark suite, MET expansion, object detection.

---

## Product scope (MVP)

Three surfaces, nothing else:

1. **Scan** — upload or camera capture of a painting.
2. **Result** — the cascade output (matched artwork + metadata, OR predicted labels, OR "uncertain") plus a RAG-generated curator's note.
3. **About** — engineering + dataset writeup (the home for all the ML work).

---

## Core flow (the cascade)

```
image
  → CLIP embedding (ONNX Runtime, CPU)
  → pgvector cosine ANN search over 81K embeddings
        Tier 1  score ≥ high      → exact match: return artwork + metadata
        Tier 2  mid score         → "likely same artist/period": return neighbors as context
  → if no confident retrieval → MLP heads
        Tier 3  confident labels  → return predicted style/genre/artist
        Tier 4  low confidence    → "uncertain"
  → labels/metadata → RAG (retrieve grounding docs + generate) → curator's note
  → response
```

---

## Build phases

### Phase A — Backend service (containerized from day one)
- [ ] FastAPI app. Endpoints: `POST /scan` (image → cascade result), `GET /health`.
- [ ] CLIP inference via **ONNX Runtime (CPU)** — this is what makes serving cheap.
- [ ] pgvector retrieval query + MLP head inference + cascade logic with tuned thresholds.
- [ ] Write the **Dockerfile now** and develop inside the container. *(Containerization = gap-closing evidence #1.)*
- [ ] Structured JSON logging + per-request timing (feeds your latency numbers).

### Phase B — RAG (the curator's note)
- [ ] Corpus: short artist/movement descriptions (Wikipedia intros + WikiArt movement blurbs). Embed into a pgvector table.
- [ ] Retrieve by predicted labels/metadata → prompt an LLM to write a 2–4 sentence grounded note.
- [ ] *Lean fallback if time is tight:* labels → LLM with structured context (no retrieval), add grounding after. But since RAG is a talking point and you've shipped one before, aim for the grounded version.

### Phase C — Frontend (minimal)
- [ ] Claude Design → React.
- [ ] **Scan** page: file upload + `getUserMedia` camera.
- [ ] **Result** page: render each cascade tier differently (match card vs label chips vs uncertain state) + loading + error states.
- [ ] **About** page.
- [ ] Talk to backend via `fetch`; enforce upload size limit client-side.

### Phase D — Deploy (the whole point)
- [ ] Managed **Postgres + pgvector** (Supabase or Neon). Load the 81K embeddings + metadata + RAG corpus.
- [ ] Backend container → **Fly.io or Render** (deploy your Dockerfile).
- [ ] Frontend → **Vercel**.
- [ ] **GitHub Actions:** on push → build image → smoke test → deploy. *(CI/CD = gap-closing evidence #2.)*
- [ ] Prod hardening: env/secrets via platform, upload size cap, error states, health check, request logging.
- [ ] **Benchmark:** measure p50/p99 latency of `/scan` on the deployed CPU instance → put numbers on About.

---

## About page (where the ML work lives)

- **Dataset:** size, label distributions, class imbalance and how it was handled.
- **Design decisions:** frozen CLIP + MLP heads (vs full fine-tune); pgvector over a dedicated vector store; ONNX-on-CPU serving; the cascade thresholds.
- **Results:** accuracy table + the "artist beats style" finding.
- **Latency:** p50/p99 from the deployed service.

---

## Future work (descoped)

- Alternate backbones (DINO / ResNet50 / ViT) comparison.
- Robustness benchmark (blur/crop/tilt/color-shift/OOD), confusion matrices, per-class F1.
- MET dataset expansion; object detection / crop-before-scan.
- Gallery; minigame.