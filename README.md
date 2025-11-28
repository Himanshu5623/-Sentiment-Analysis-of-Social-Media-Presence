Sentiment Analysis Research Project (YouTube + Reddit +
Twitter)
Lovely Professional University — B.Tech Project
Author: Himanshu Agnihotri
Project Type: Research / Academic Proof-of-Concept
Purpose: Large-scale sentiment analysis using social media comments
📌 Overview
This project builds a complete sentiment analysis pipeline using comments from Reddit, YouTube,
and Twitter. It combines:
• Automated data collection (Reddit API, YouTube API, Twitter API)
• Text preprocessing and cleaning
• Human-in-the-loop labeling system
• Machine learning models: Logistic Regression, Naive Bayes, Oversampling, Balanced
Weights
• Streamlit dashboards for visual analysis
• Batch sentiment analysis on newly collected data
• End-to-end MLOps-style training pipeline
This project is strictly read-only—no posting, voting, or messaging.
📊 High-Level Architecture
(Attaching a PNG diagram below — download link appears later.)
┌─────────────┐ ┌──────────────────┐
│ Reddit API │ │ YouTube API │
│ Twitter API │ │ (optional) │
└──────┬──────┘ └─────────┬────────┘
│ │
▼ ▼
┌────────────────────────────────────────┐ │ Data Ingestion (Python Scripts) │
│ - reddit_fetch.py │
│ - youtube_fetch.py │
│ - twitter_fetch.py │ └────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────┐ │ Data Preprocessing Pipeline │
│ text cleaning, normalization, labels │ └────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────┐ │ ML Training (scikit-learn) │
│ - TF-IDF Vectorizer │
│ - Logistic Regression │
│ - Naive Bayes │
│ - Oversampling / Balancing │ └────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────┐ │ Evaluation, Metrics, Reports │
│ ROC, Confusion Matrix, F1, Accuracy │ └────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────┐ │ Streamlit Dashboard + Batch Inference │
│ predictions_*.parquet files │ └────────────────────────────────────────┘
🎯 Goals
• Build a robust multi-source sentiment dataset (5,000+ examples)
• Achieve 80–88% accuracy using improved ML pipelines
• Provide visual dashboards + downloadable predictions
• Publish the final dataset + report for academic use
🧩 Features
✔ Data Collection
• Reddit (PRAW)
• YouTube comments
• Twitter/X scraping
✔ Preprocessing
• Cleaning
• Normalization
• Lemmatization
• Language filtering
✔ Modeling
Model Purpose
Logistic Regression Strong baseline model
Logistic Regression (Weighted) Best performing model
Logistic Regression
(Oversample) Handles imbalance
Naive Bayes Fast / interpretable
baseline
✔ Evaluation
• Accuracy
• Precision/Recall/F1
• ROC Curves
• Confusion Matrices
✔ Batch Prediction
Generates files like:
predictions_20251127_230940.parquet
predictions_nb_20251128_153020.parquet
✔ Streamlit Dashboard
Interactive filters:
• Source
• Sentiment
• Confidence
• Time range
• Text viewer
🔐 Reddit API Use (Important for Reviewers)
• App is registered as a script app.
• Only performs read-only actions.
• Used for academic research.
• Rate limits respected.
• Will not scrape private content or deleted content.
• Will not post/vote/comment.
• Will honor moderator and Reddit requests to remove data.
📁 Repository Structure
sentiment-analysis-project/ │ ├── src/
│ ├── data_collection/
│ │ └── reddit_fetch.py
│ │ └── youtube_fetch.py
│ │ └── twitter_fetch.py
│ ├── preprocessing/
│ ├── train/
│ ├── streamlit_poc_improved.py
│ ├── data/
│ ├── raw/
│ ├── processed/
│ └── outputs/
│ ├── models/
│ ├── logreg_sentiment.joblib
│ ├── tfidf_vectorizer.joblib
│ ├── naive_bayes.joblib
│ └── README.md
