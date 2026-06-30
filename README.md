# Phase 0: Dataset Analysis

- [x]  Finding the dataset
- [x]  Analyze the dataset (tally the labels)
- [x]  Cleaning the dataset
- [x]  Split the dataset

### Dataset

We will be using the Huggingface Wikiart dataset (~81, 000 works with artist, genre, and style labels). https://huggingface.co/datasets/huggan/wikiart

Before training, first check

- Identify all the possible artist, genre, and style labels
- Output the number/distribution between all the labels. Sort and display all the labels as well as the number of works with that label
- Identify biases/categories with larger weights

Splitting data into 80 : 10 : 10 for train, validation, test → stratified

# Phase 1: Which Vision Architecture Best Understand Paintings

- [ ]  Build CLIP Top-k retrieval pipeline
- [ ]  Build CLIP classification pipeline
- [ ]  Evaluate between Top-k performance versus classification performance (for both exact artwork retrieval or related works - ex. artist is in databse but not the exact piece)
    - [ ]  Evaluate top1 vs top 5 matches, latency, exact retrieval vs closely related works not in database/degraded works, etc
- [ ]  Narrower testing variations for each
    - [ ]  Ex. Top-k with different image models like Dino, Classification with ViT or CNN instead
- [ ]  Loss/Accuracy curves, overfitting/underfitting/convergence speed, training/validation loss and accuracy

Ideally, we want

exact match→retrieve artwork metadata

no confident match→predict style/artist/period labels

OOD/low confidence→say uncertain

# Top-K Retrieval vs Classification Labels

First, we want to test 2 different type of pipelines to see which produces the better results in terms of exact matches, relevant context retrieval, and such. We will test out one type of model from each as a benchmark, then if there is a clear advantage in one, we will test more nuanced variations within that path (ex. specific models/dimension sizes, different top-k values, etc.)

Path A: CLIP ViT-B/32 → cosine search → top-10, cosine-weighted aggregation
Path B: CLIP ViT-B/32 → trained MLP heads → label aggregation

Same testing/training data and using similar embedding model to make it fair

### Path A: Photo → CLIP Embeddings → Cosine Search → Aggregate Metadata from Top-k Image matches for RAG

This is more **retrieval** rather than training. Directly embeds and matches closest embeddings. The metadata directly returned, then high confidence results passed into RAG

We will begin by testing CLIP embeddings/DINO

### Path B: Photo → CLIP Embeddings → Classification MLP Head → predict style/genre/artist textual labels → Aggregate Labels for RAG

This is more **training** since we are training a model to predict labels from images. The labels are then used as the “query” for RAG

We will begin by testing CLIP embeddings + MLP head

### Retrieval vs Labeling Evaluation Tests

**Test Sets**

1. Known artworks in DB
2. Degraded known artworks (blurs, tilts, crops, direct pics (like taken in museum with walls/other backgrounds etc.), color shift, angles, etc.)
3. Stylistically related but not in DB (Ex. monet works exist in the database, but testing monet works that wasn’t directly in DB)
4. OOD (pictures whose style/artist are just not in the database), ex. modern photography? sketches? etc.)

**Evaluation Metrics**

1. Exact Retrieval (Top - 1), did it make the correct guess
2. Top K - Is the correct label/work within the top k?
3. How do scores distribute throughout those datasets?
4. Confidence Thresholds? (want to see what labels each path *produces* when there's no good match)
5. Latency

## Top-K Retrieval Expansion (If Clear Winner)

**Objective:** Evaluate whether different models/top-k options are better at retrieving the correct labels/information about an artwork. 

### Models

**CLIP**

**DINO**

Other good image models…?

### Variations

Return either Top-K (Different k values, top 3 vs 5 vs 10 vs 20?) or any past certain confidence threshold/then top categories?

Aggregation strategies: flat vote, cosine-weighted, threshold-filtered

### Evaluation

Retrieving exact matches (giving artworks that exist in the dataset given but might add variations like blur, tile, image in a frame but taken from further away, or smthn)

Predicing images that are still clearly stylistically from the same artist/period but not in the dataset

Predicting images that are NOT very similar (hallucination?)

## Classification Labels Expansion (If Clear Winner)

**Objective:** Evaluate whether foundation-model representations (CLIP) or fully fine-tuned vision models (ResNet/ViT) are better at predicting artistic characteristics from paintings.

### Models

**CLIP** 

- Zero shot (Classification via prompt matching? Mostly for foundational baseline)
- CLIP + Linear Head (Frozen CLIP → Embedding → Linear Layer)
- CLIP + MLP (Frozen CLIP → Embedding → MLP)

**ResNet50** → Cnn baseline. Pretrained, then fine-tune 

**ViT →** Fine tuned vision transformer

### Evaluation

- Style Classification (Accuracy, F1, Top3/5 Accuracy)
- Genre Classification (Accuracy, F1, Top3/5 Accuracy)
- Artist Classification (Accuracy, F1, Top3/5 Accuracy)
- Check for overfitting/underfitting/convergence speed, Track:
    - Training loss
    - Validation loss
    - Training accuracy
    - Validation accuracy
    
    Plots:
    
    - Loss curves
    - Accuracy curves
- Robustness Benchmark (Blurs, crop, lighting changes, perspective distortion, tilts, etc.)
    - Report relative accuracy drop for each corruption type
- Error analysis of confusion matrix (which artist/styles/genres are confused most often?)
- Per-Class Performance (F1) Analysis

#### **Scope Limitations**

- Paintings only (Leave out artifacts and sculptures)
- Leave out matching/retrieval (dont worry about returning the exact work first, we first just produce correct labels and classifications)
- Limit to just the WikiArt Huggingface Dataset
- Might try different scaling variations later (start with CLIP ViT-B/32 and test larger ones later)
- Later will attempt object detection → Scaling/cropping image before scanning

# Phase 2: RAG, Query Rewriting, Expanding Datasets and Refining the Model

Rewriting the labels retrieved from the models into a good RAG input

Biggest part of phase 2: Building the RAG portion

Consider experimenting with MET data instead (?)

# Phase 3: System Design and Full-Stack

- [ ]  Claude Design prototype
- [ ]  Convert to React
- [ ]  Build connecting pieces
    - [ ]  Database
    - [ ]  API to backend
- [ ]  Build javascript pieces that connects static frontend (working camera, check that interactions sends api signals to backend)
- [ ]  Production Ready things
    - [ ]  Upload size limits
    - [ ]  Error states
    - [ ]  Loading states
    - [ ]  API logging