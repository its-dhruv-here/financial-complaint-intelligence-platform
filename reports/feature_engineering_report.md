# Feature Engineering Experiments

| Experiment | Vocab Size | Sparsity | Fit Time (s) | Inference Time (s) |
|---|---|---|---|---|
| A. TF-IDF Unigrams | 103,625 | 99.9121% | 45.84 | 0.223 |
| B. TF-IDF Uni+Bigrams | 2,898,107 | 99.9915% | 155.5 | 0.709 |
| C. TF-IDF max-10k features | 10,000 | 99.1019% | 32.21 | 0.114 |
| D. TF-IDF + stopword removal | 103,311 | 99.9471% | 28.21 | 0.089 |
| E. TF-IDF + sublinear TF | 103,625 | 99.9121% | 29.36 | 0.085 |
