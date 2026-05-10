# Neural Chunking: BiLSTM vs Transformer

Sequence labelling for shallow parsing in PyTorch. A bidirectional LSTM is compared against a Transformer encoder built from scratch (no `nn.Transformer`, no `transformers` library). Each model predicts one of 17 CoNLL-style chunk tags per token.

## Task

Given a sentence as a sequence of words, predict the chunk tag for each word:

```
Confidence  B-NP
in          B-PP
the         B-NP
pound       I-NP
is          B-VP
widely      I-VP
expected    I-VP
```

Chunks group consecutive words into shallow phrases (noun phrases, verb phrases, prepositional phrases, and so on). It is a common preprocessing step for information extraction, question answering, and other downstream NLP.

The 17 tags follow the BIO scheme: `B-XX` marks the beginning of a chunk, `I-XX` marks a token inside an ongoing chunk, and `O` marks tokens outside any chunk.

## Dataset

CoNLL-2000 style chunking data, ~8,900 sentences, distributed via the COMP534 module at the University of Liverpool. Place the file at `data/chunk_dataset.txt` (the path is configurable via `DATA_PATH` in the notebook). The dataset is not included in this repository.

## Architectures

Both models share the same head: an embedding lookup, a context encoder, and a per-token linear classifier over the 17 tags. They differ only in the context encoder.

**BiLSTM tagger**

| Component | Setting |
|---|---|
| Embedding | 128-dim, padding-aware |
| Encoder | 1-layer bidirectional LSTM, hidden=128 per direction |
| Dropout | 0.3 |
| Head | Linear(256, 18) |

**Transformer encoder tagger** (from scratch)

| Component | Setting |
|---|---|
| Embedding | 128-dim, padding-aware |
| Position | Sinusoidal positional encoding |
| Encoder | 3 stacked encoder blocks |
| Attention | 4-head scaled dot-product self-attention, padding-masked |
| Feed-forward | Linear(128, 256) → ReLU → Linear(256, 128) |
| Block | Residual + LayerNorm around attention and feed-forward |
| Dropout | 0.1 |
| Head | Linear(128, 18) |

Both models are roughly matched on parameter count for a fair comparison. The Transformer is implemented entirely with `torch.nn` primitives (no `nn.Transformer`, no `nn.MultiheadAttention`, no `transformers`).

## Methodology

- **Split:** 80 / 10 / 10 train / val / test at the sentence level, seeded.
- **Vocabulary:** built from the training set only. `<PAD>` and `<UNK>` reserved. Tags use a `<TAG_PAD>` slot so padded positions can be ignored by the loss.
- **Padding:** each batch is padded to its own longest sentence, not corpus max.
- **Loss / optimiser:** `CrossEntropyLoss(ignore_index=TAG_PAD)` and Adam at 1e-3.
- **Epochs:** 10 for each model.
- **Best model selection:** by validation macro-F1, not accuracy. Token accuracy is dominated by frequent tags (I-NP, B-NP, O together account for ~62% of all tokens) and underweights performance on rare tags such as B-INTJ and I-CONJP. Macro-F1 averages F1 across all 17 tags equally.
- **Test metrics:** accuracy, macro-F1, weighted-F1, per-class classification report, and a 17×17 confusion matrix.
- **Reproducibility:** seed = 72 across `random`, `numpy`, and `torch`.

## Setup

```bash
pip install -r requirements.txt
```

Tested on Python 3.10+ with PyTorch 2.x. GPU optional but recommended.

## Run

```bash
jupyter notebook nlp_chunking.ipynb
```

Run all cells top to bottom. Training the two models takes a few minutes on a modern GPU and longer on CPU.

## Project structure

```
.
├── nlp_chunking.ipynb     main notebook
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── data/
    └── chunk_dataset.txt  not tracked; place dataset here
```

## License

MIT. See `LICENSE`.
