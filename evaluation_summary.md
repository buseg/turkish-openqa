# Evaluation Summary

This file is generated from the current JSON metric files by `scripts/summarize_evaluation.py`.

## Final Ablation Summary

Reader metrics are 200-example answerable FSMODQA smoke checks. Retrieval metrics are full-dev unless the setting name says otherwise.

| Ablation | System | Retrieval Ref. (R@K) | Contexts | EM | F1 | Correctness | Faithfulness | Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | BM25 top-5 + FSMODQA | 15.67 | 5 | 9.00 | 15.17 | 15.32 | 72.84 | Main fixed-retrieval baseline. |
| More Contexts | BM25 top-25 + FSMODQA | 27.49 | 25 | 10.00 | 14.80 | 15.16 | 84.74 | More evidence improves EM, but not F1. |
| Adaptive Retrieval | Retrieval-aware 50/50/100 + FSMODQA | 34.81 | 76.86 avg | 10.00 | 15.42 | 15.95 | 81.74 | Best tested reader F1 with fewer contexts than fixed top-100. |
| Question Rewriting | Keyword rewrite top-5 + FSMODQA | 15.74 | 5 | 7.00 | 10.55 | 11.33 | 78.65 | Small retrieval gain does not transfer to reader quality. |
| Selective Rewriting | Selective keyword rewrite top-5 + FSMODQA | 15.67 | 5 | 7.00 | 10.91 | 11.45 | 79.39 | Selection helps slightly over pure rewrite, but still below baseline. |
| Knowledge Selector | TF-IDF selector hybrid top-5 + FSMODQA | 16.53 | 5 | 9.00 | 14.85 | 15.24 | 72.11 | Close to baseline, but no reader gain. |
| Neural Selector | BERTurk selector hybrid top-5 + FSMODQA | 23.50 | 5 | 8.50 | 13.95 | 14.32 | 75.35 | Improves held-out selector MRR, but not reader F1 yet. |
| Oracle Upper Bound | Oracle top-5 + FSMODQA | - | 5 | 12.50 | 19.31 | 20.16 | 73.83 | Not deployable; shows retrieval still bottlenecks QA. |

## Retrieval And Selection

| Setting | Examples | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR | Avg ctx | Saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 top-100 | 2910 | 7.80 | 15.67 | 19.90 | 27.49 | 31.79 | 37.29 | 11.94 | - | - |
| Adaptive oracle 10/50/100 | 2910 | 7.80 | 15.67 | 19.90 | 27.49 | 31.79 | 37.29 | 11.94 | 76.15 | 23.85 |
| Adaptive question-only 25/50/100 | 2910 | 7.80 | 15.67 | 19.90 | 27.49 | 29.97 | 32.51 | 11.85 | 63.39 | 36.61 |
| Adaptive retrieval-aware 25/50/100 | 2910 | 7.80 | 15.67 | 19.90 | 27.49 | 29.79 | 32.82 | 11.85 | 67.12 | 32.88 |
| Adaptive retrieval-aware 50/50/100 | 2910 | 7.80 | 15.67 | 19.90 | 27.49 | 31.79 | 34.81 | 11.91 | 76.86 | 23.14 |
| TF-IDF selector hybrid top-5 | 2910 | 8.04 | 16.53 | - | - | - | - | 11.18 | - | - |
| TF-IDF selector hybrid top-25 | 2910 | 8.04 | 16.53 | 20.34 | 27.42 | - | - | 12.12 | - | - |
| BERTurk selector smoke top-5/50ex | 50 | 16.00 | 30.00 | - | - | - | - | 21.17 | - | - |
| BERTurk selector medium hybrid top-5/200ex | 200 | 14.00 | 23.50 | - | - | - | - | 17.45 | - | - |

## Question Rewriting

These are retrieval-only full-dev checks; 200-example reader checks are reported in the Reader / QA table.

| Setting | Examples | R@1 | R@5 | R@10 | R@20 | R@25 | R@50 | R@100 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 original | 2910 | 7.80 | 15.67 | 19.90 | - | 27.49 | 31.79 | 37.29 | 11.94 |
| Rule rewrite: broad | 2910 | 7.11 | 14.54 | 18.49 | 23.47 | 25.12 | 30.38 | 36.15 | 10.95 |
| Rule rewrite: keyword | 2910 | 7.63 | 15.74 | 20.14 | 25.67 | 27.42 | 31.99 | 37.66 | 11.87 |
| Selective keyword rewrite | 2910 | 7.84 | 15.67 | 20.03 | 25.77 | 27.73 | 31.89 | 37.46 | 11.98 |

## Adaptive Classifier

| Setting | Accuracy | Macro F1 | Weighted F1 | Easy F1 | Medium F1 | Hard F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Question-only adaptive classifier | 51.70 | 44.12 | 53.11 | 63.26 | 22.43 | 46.69 |
| Retrieval-aware adaptive classifier | 65.80 | 52.38 | 64.81 | 77.87 | 20.28 | 59.00 |

## Reader / QA

These are 200-example FSMODQA reader checks, so they are smoke/evidence runs rather than final full-dev scores.

| Reader Input | Examples | EM | F1 | Correctness Recall | Faithfulness K-Prec | Evidence Contains Pred | Refusal | Contains Gold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 top-5 | 200 | 9.00 | 15.17 | 15.32 | 72.84 | 77.00 | 0.00 | 11.50 |
| BM25 top-25 | 200 | 10.00 | 14.80 | 15.16 | 84.74 | 80.50 | 0.00 | 12.00 |
| Keyword rewrite top-5 | 200 | 7.00 | 10.55 | 11.33 | 78.65 | 78.50 | 0.00 | 10.00 |
| Selective keyword rewrite top-5 | 200 | 7.00 | 10.91 | 11.45 | 79.39 | 79.00 | 0.00 | 9.50 |
| Adaptive retrieval-aware 50/50/100 | 200 | 10.00 | 15.42 | 15.95 | 81.74 | 75.50 | 0.00 | 13.50 |
| Oracle top-5 | 200 | 12.50 | 19.31 | 20.16 | 73.83 | 77.50 | 0.00 | 17.00 |
| TF-IDF selector hybrid top-5 | 200 | 9.00 | 14.85 | 15.24 | 72.11 | 74.50 | 0.00 | 11.50 |
| BERTurk selector medium hybrid top-5 | 200 | 8.50 | 13.95 | 14.32 | 75.35 | 77.00 | 0.00 | 10.00 |

## Refusal / Unanswerable QA

| Setting | Examples | Answerable | Unanswerable | Answerable F1 | Faithfulness K-Prec | Refusal Rate | Correct Refusal | Answered Unanswerable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 top-5 mixed 50 answerable / 50 unanswerable | 100 | 50 | 50 | 24.56 | 82.80 | 0.00 | 0.00 | 100.00 |

## Knowledge Selector Held-Out Split

| Setting | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 34.40 | 54.10 | 59.50 | 66.50 | 72.70 | 78.70 | 43.31 |
| TF-IDF selector only | 27.70 | 55.70 | 65.10 | 73.50 | 77.50 | 78.70 | 40.46 |
| TF-IDF selector/BM25 hybrid | 40.70 | 61.20 | 67.90 | 75.10 | 77.30 | 78.70 | 50.10 |

## Neural Knowledge Selector Medium Run

| Setting | R@1 | R@5 | R@10 | R@25 | R@50 | R@100 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 held-out | 47.00 | 63.00 | 69.00 | 70.00 | 77.00 | 80.00 | 53.61 |
| BERTurk selector only | 33.00 | 64.00 | 67.00 | 75.00 | 76.00 | 80.00 | 44.55 |
| Hybrid weight 0.5 | 51.00 | 65.00 | 71.00 | 74.00 | 77.00 | 80.00 | 57.30 |

## Current Reading

- BM25 top-100 is still the main full-dev retrieval baseline.
- Oracle adaptive retrieval shows the efficiency upper bound: same retrieval recall with fewer contexts.
- Retrieval-aware adaptive classification improves classifier F1, but still loses some Recall@100 compared with fixed top-100.
- Keyword question rewriting gives a small retrieval gain at higher K, but it hurts the 200-example reader F1 in the current top-5 setup.
- Reader/QA F1 has been checked on 200-example smoke runs, including adaptive retrieval-aware 50/50/100.
- The current FSMODQA reader does not refuse on the mixed unanswerable smoke set; it answers all unanswerable questions.
- The medium BERTurk selector improves held-out selector MRR but did not improve 200-example reader F1 in its current top-5 setup.
