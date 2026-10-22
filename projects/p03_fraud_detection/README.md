# P03 Fraud detection

**Question:** A fraud team can only review a limited number of transactions a day. Which ones should they look at first?

**Data:** 20,000 synthetic card transactions, 2% fraudulent, with amount, hour, distance from home, merchant category, foreign flag, recent transaction count and card-present flag.

**Approach**
1. Engineer features such as log amount and a night-time flag.
2. Handle class imbalance with class weights rather than resampling.
3. Compare class-weighted logistic regression with a random forest.
4. Judge models by average precision and precision at k, because with 2% fraud a model that predicts "not fraud" for everything is 98% accurate and useless.
5. Translate scores into a review-budget table: precision, recall and fraud value caught for 50, 100, 200 and 400 reviews.

**Result:** Logistic regression scores best, with average precision 0.49 (about 24x the 2% base rate). Reviewing the top 50 alerts is 78% precise; reviewing 400 catches 71% of all fraud.

**Run:** `python -m projects.p03_fraud_detection.analysis`
