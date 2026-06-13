# Model Comparison Report

This report summarizes the performance of models evaluated on the full 383,548-record dataset.

## Summary

| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 | Train Time (s) |
|---|---|---|---|---|---|---|
| Dummy (Most-Frequent) | 0.2409 | 0.0580 | 0.2409 | 0.0228 | 0.0935 | 0.19 |
| Multinomial Naive Bayes | 0.6934 | 0.6889 | 0.6934 | 0.4306 | 0.6679 | 1.40 |
| Logistic Regression | 0.7127 | 0.7493 | 0.7127 | 0.5665 | 0.7227 | 155.11 |
| **Linear SVM** | **0.7277** | **0.7419** | **0.7277** | **0.5591** | **0.7327** | **272.82** |

## Conclusion

The **Linear SVM** model outperformed all other baselines on the full dataset, achieving the highest Accuracy (72.77%) and Weighted F1 (73.27%). Logistic Regression was a close second, while Multinomial Naive Bayes struggled significantly with Macro F1 on minority classes.
