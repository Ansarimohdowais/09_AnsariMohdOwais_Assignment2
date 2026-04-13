# Sentiment Analysis on Asha Bhosle Tweets

This repository contains a small Python project for tweet sentiment classification on the topic **Asha Bhosle**.

## What is inside
- `data/asha_bhosle_tweets.csv` - 100 manually labeled tweet-like samples
- `notebook/asha_bhosle_sentiment_analysis.ipynb` - notebook version of the workflow
- `main.py` - runnable Python script
- `results/` - plots, metrics, and confusion matrices
- `reports/final_report.pdf` - final assignment report

## Assignment requirements covered
- Topic: **Asha Bhosle**
- Dataset: 100 labeled samples
- Split: 80 train / 20 test
- Classifiers: Naive Bayes, SVM, Logistic Regression
- Metrics: Precision and Recall
- Visualizations: label distribution, model comparison, confusion matrices

## Important note
This project uses a **small sample dataset** of tweet-like texts for a complete runnable demo. If you want to use real tweets, replace the CSV with tweets collected through the X/Twitter API or a licensed dataset, then rerun `main.py`.

## How to run
```bash
pip install -r requirements.txt
python main.py
```

## Results summary
| Model | Precision (macro) | Recall (macro) |
|---|---:|---:|
| Naive Bayes | 0.9583 | 0.9524 |
| SVM | 1.0000 | 1.0000 |
| Logistic Regression | 1.0000 | 1.0000 |

## Suggested 3 commits
1. **Commit 1:** Add dataset, notebook, preprocessing, and baseline training code.
2. **Commit 2:** Add evaluation, plots, metrics export, and model comparison.
3. **Commit 3:** Add README, final report PDF, and polish the repository structure.

## Student details
- Name: __________________
- Roll No: __________________
- UIN: __________________
- Year: TE-AIDS
