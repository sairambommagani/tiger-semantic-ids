# Generative retrieval with semantic IDs (TIGER) on Amazon Reviews 2023

A TIGER-style recommender (Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023) on the
Amazon Reviews 2023 categories Beauty and Personal Care, Sports and Outdoors, and Toys and Games. Instead of scoring
every item, the model **generates the next item's ID token by token**, where the ID is a *semantic ID* learned from
the item's content, so similar items share ID prefixes.

**Research prototype.**

> Results (20,000 held-out users per category, full ranking): TIGER Recall@10 **0.0160** on Beauty (popularity 0.0086)
> and **0.0143** on Sports (popularity 0.0053); SASRec 0.0161 and 0.0120. Toys (TIGER 1 epoch only): TIGER 0.0080,
> SASRec 0.0098, popularity 0.0049. Details in section 4.

## 1. How it works

```
item text (title, brand, categories, price, features)
   -> Sentence-T5 embedding (768-d)
   -> RQ-VAE: 3 residual codebooks of 256 codewords      -> (c1, c2, c3)
   -> + collision token for items with the same 3 codes  -> semantic ID (c1, c2, c3, c4), unique per item

user history (last 20 items) -> tokens: [user] c1 c2 c3 c4  c1 c2 c3 c4 ...
   -> T5 encoder-decoder (4 + 4 layers, d 128, 6 heads)
   -> beam search over 4 tokens, constrained by a trie of valid IDs -> top-10 items
```

Item embeddings are centred and scaled per dimension before the RQ-VAE: sentence embeddings all point in a narrow
cone (cosine 0.7-0.9 between unrelated items), and without this step the codebooks collapse onto a single code.

Why it matters: the item "vocabulary" is a few codebooks instead of one embedding per item, similar items share
parameters through shared codes, and an item never seen in training still has a meaningful ID from its text.

## 2. Data and protocol

- Amazon Reviews 2023 (McAuley Lab) **5-core** rating files and item metadata, one category per run.
- **Full 5-core data** for each category (Beauty 5-core: 730K users, 208K items, 6.6M ratings). Training examples
  (every prefix of every training sequence, millions) are built on the fly, and a checkpoint is saved after every
  epoch so a Colab disconnect does not lose the run.
- Each user's items are sorted by time: last item = test target, second last = validation target, the rest =
  training (every prefix is a training example), at most 20 items of history.
- **Full ranking**: every model ranks all items. Beam-search generation for every user would take hours, so
  Recall@5/10 and NDCG@5/10 are measured on **20,000 random held-out users** per category (the same users for every
  model); 95% of the metric's sampling error is then about +-0.005 on Recall@10.
- **Cold items**: test targets that never appear in any training sequence; reported separately.

## 3. Models compared

| Model | What it is |
|---|---|
| Popularity | the most frequent training items for everyone |
| SASRec | self-attentive sequential recommender: one learned embedding per item ID (Kang & McAuley 2018) |
| **TIGER** | the generative retriever over semantic IDs (deployed) |
| TIGER, random IDs (optional) | same model with random codes of the same shape: shows what the *semantic* part adds |

Semantic-ID checks: codebook usage per level, share of items with a unique code before the collision token, the
largest collision, and **category purity**: how often items that share the first code (or first two codes) have the
same leaf category, compared with random IDs.

## 4. Results

Full 5-core data, all users for training, 20,000 random held-out users per category for the test (same users for
every model), full ranking over all items. Trained on a free Colab T4 with fp16 mixed precision.

| | Beauty and Personal Care | Sports and Outdoors | Toys and Games |
|---|---|---|---|
| Users / items / ratings | 729,576 / 207,649 / 6.62M | 409,772 / 156,235 / 3.47M | 432,264 / 162,035 / 3.86M |
| Training prefixes | 4,435,713 | 2,242,704 | 2,565,094 |
| TIGER epochs (batch 1024) | 2 (55 min) | 2 (28 min) | 1 (16 min) |

**Semantic IDs (RQ-VAE, 3 x 256 codes + collision token)**

| | Beauty | Sports | Toys |
|---|---|---|---|
| Codebook usage per level | 100 / 100 / 100% | 100 / 100 / 100% | 100 / 100 / 100% |
| Items with a unique code before the collision token | 86.4% | 82.9% | 87.4% |
| Largest group sharing one code | 44 items | 36 items | 34 items |
| Items sharing the 1st code with the same leaf category (random IDs) | **60.8%** (3.1%) | **45.1%** (2.7%) | **48.2%** (8.8%) |
| Items sharing the first 2 codes with the same leaf category (random IDs) | **78.4%** (28.6%) | **76.0%** (32.3%) | **70.6%** (33.8%) |

**Recommendation (test, 20,000 users)**

| Category | Model | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| Beauty | **TIGER** | **0.0100** | **0.0160** | **0.0066** | **0.0085** |
| Beauty | Popularity | 0.0062 | 0.0086 | 0.0042 | 0.0050 |
| Beauty | SASRec | 0.0091 | **0.0161** | 0.0060 | 0.0083 |
| Sports | **TIGER** | **0.0095** | **0.0143** | **0.0064** | **0.0079** |
| Sports | Popularity | 0.0031 | 0.0053 | 0.0019 | 0.0026 |
| Sports | SASRec | 0.0070 | 0.0120 | 0.0044 | 0.0060 |
| Toys | TIGER (1 epoch) | 0.0055 | 0.0080 | **0.0037** | 0.0045 |
| Toys | Popularity | 0.0020 | 0.0049 | 0.0012 | 0.0021 |
| Toys | SASRec | **0.0056** | **0.0098** | 0.0034 | **0.0047** |

What the numbers say:

1. **The semantic IDs carry meaning.** Every codeword is used, and items that share the first code are in the same
   leaf category 45-61% of the time against about 3% for random IDs of the same shape: the RQ-VAE turns item text
   into a coarse-to-fine hierarchy.
2. **TIGER beats popularity by 1.9x (Beauty) and 2.7x (Sports) on Recall@10**, on a full ranking over 156K-208K
   items. **Against SASRec** (one learned embedding per item ID): TIGER is better on Sports (Recall@10 +19%, NDCG@10
   +32%) and level on Beauty (Recall@10 0.0160 vs 0.0161, NDCG@10 0.0085 vs 0.0083; with 20,000 test users the
   sampling error of Recall@10 is about +-0.002, so this is a tie). TIGER does this with a vocabulary of about 1,000
   code tokens instead of one embedding per item (156K-208K), and it was still improving when training stopped. Absolute values are low because the 2023 catalogues are about 13x larger than the 2014 benchmark used in the
   paper (12K items for Beauty); they are not comparable with the paper's numbers.
   **Toys** had only 1 TIGER epoch (time budget): TIGER is 1.6x popularity but below SASRec on Recall@10 (0.0080 vs
   0.0098) and level on NDCG (0.0045 vs 0.0047). After one epoch the other categories were also behind their final
   value (validation Recall@10 after epoch 1 -> 2: Beauty 0.012 -> 0.021, Sports 0.0145 -> 0.0160), so Toys should be
   re-run with 2 epochs before drawing a conclusion.
3. **TIGER learns the ID grammar**: without the trie constraint only 0.04% (Sports) and 0.0% (Beauty) of the
   generated IDs are invalid after 2 epochs (Toys, 1 epoch: 1.6%), so the constraint is a safety net, not a crutch.
4. **Still improving**: Beauty validation Recall@10 went from 0.0120 after epoch 1 to 0.0205 after epoch 2; more
   epochs (the paper trains about 200K steps) would raise all numbers.
5. **Cold items are not measurable on 5-core data**: 5-core guarantees every item at least 5 ratings, so only
   0.1-0.4% of test targets are unseen in training (about 30-70 of 20,000 users); the cold-item advantage of semantic
   IDs needs a sparser setting (a 30K-user sample had 23% cold targets).
6. **SASRec**: the first runs gave near-zero scores because of a bug (a key-padding mask on left-padded sequences
   produced NaN scores in evaluation mode). It was reproduced on synthetic data, fixed (`tiger/baselines.py`) and
   re-run with the same split and the same 20,000 users (`tiger/rerun_sasrec.py`, results in `sasrec_fixed.json`);
   TIGER and popularity results are not affected. SASRec's validation Recall@10 peaked early (Sports: 0.014 at epoch
   5, then 0.009) - it overfits the per-item embeddings; the best validation epoch is the one reported.

### Result files (proof for every number above)

`results/<category>/` holds, for each trained category:

| File | What it contains |
|---|---|
| `metrics.json` | all test and validation results, per-epoch training history, semantic-ID statistics, data summary |
| `version.json` | the model card: version name, creation time, training-data SHA-256, code hash, file hashes of the model |
| `sasrec_fixed.json` (Beauty, Sports) | SASRec re-run after the bug fix, same split and the same 20,000 users; it replaces the `sasrec` entry in that category's `metrics.json` |

The model weights themselves (about 30 MB per category) are kept out of git; any of the three can be served with
`TIGER_ARTIFACTS=<folder> python -m uvicorn tiger.api:app --port 8014`.

## 5. Training on Google Colab, FastAPI on the laptop

Colab (`notebooks/train_colab.ipynb`, one category per run): upload the code zip; the notebook downloads the
category's 5-core ratings and metadata from the McAuley Lab server (no key), runs the tests, trains, evaluates,
saves the version folder to Google Drive and downloads `artifacts_<category>.zip`.

Laptop:
```powershell
pip install -r requirements.txt
Expand-Archive artifacts_beauty.zip -DestinationPath artifacts
python -m uvicorn tiger.api:app --port 8014          # web page http://localhost:8014, API docs /docs
python -m pytest -q                                  # offline tests (synthetic data, TF-IDF embeddings)
```

| Endpoint | Input | Output |
|---|---|---|
| `POST /recommend` | `{"history": ["B0...", ...], "k": 10}` (item IDs, oldest first) | top-k items with semantic IDs, titles, whether they share the first code with the history |
| `GET /search?q=` | text | items by title / brand (to build a history) |
| `GET /items/{asin}` | | one item and its semantic ID |
| `GET /samples` | | held-out users' histories and the item they bought next |
| `GET /model-info` | | model card, semantic-ID statistics, all results |

Model versions: `tiger_v<MAJOR.MINOR.PATCH>_<category>_<YYYYMMDD>` with a model card (`version.json`: data SHA-256,
code hash, semantic-ID statistics, metrics, file hashes); files are checked before loading.

## 6. Limitations

- Test metrics on 20,000 sampled users per category (training uses all users); absolute numbers are not comparable
  with papers that use the 2014 Amazon data.
- Shorter training than the paper (2-3 epochs over all prefixes, batch 1024, vs about 200K steps); one seed.
- The API recommends only items in the trained catalogue; new items would need the RQ-VAE (saved) and the text
  encoder to get an ID.

## Files

| File | What |
|---|---|
| `tiger/data.py` | ratings, user sample, sequences, leave-last-out split, metadata, item text |
| `tiger/text.py` | Sentence-T5 embeddings (TF-IDF fallback for tests) |
| `tiger/rqvae.py` | RQ-VAE, collision token, ID statistics, category purity, random-ID ablation |
| `tiger/seq2seq.py` | vocabulary, T5 model, trie-constrained beam search, training |
| `tiger/baselines.py` | popularity, SASRec |
| `tiger/metrics.py` | Recall@k, NDCG@k |
| `tiger/train.py` | the full pipeline and saving a version |
| `tiger/infer.py`, `tiger/api.py`, `tiger/static/index.html` | inference, FastAPI service, web page |
| `notebooks/train_colab.ipynb` | Colab notebook (one category per run) |
| `scripts/make_synthetic.py`, `tests/` | synthetic data and tests |
