# P04 Customer segmentation

**Question:** What distinct groups of retail banking customers are there, and how should marketing and product teams treat each one?

**Data:** 3,000 synthetic customers with age, income, average balance, number of products, monthly transactions, share of digital activity and tenure.

**Approach**
1. Log-transform skewed money features and standardize everything so no single feature dominates distances.
2. Run K-means for k = 2 to 8 and choose k by silhouette score.
3. Profile each segment with medians and visualize clusters on a 2D PCA projection.
4. Because the data is synthetic, compare clusters with the true segments using the adjusted Rand index.

**Result:** Silhouette analysis picks three segments:
- **Digital everyday bankers** (about 2,000 customers): younger, modest balances, high transaction counts, mostly digital. Good fit for app features and card products.
- **Retired savers** (about 490): older, long tenure, large balances, low digital use. Good fit for savings and advice delivered through branches.
- **Affluent multi-product** (about 475): high income and balances, five products. Good fit for wealth management and premium cards.

The data was generated with four segments, but young digital and mass-market customers overlap enough that K-means merges them. That is a useful reminder that the "right" number of clusters is a business decision, not only a statistical one.

**Run:** `python -m projects.p04_customer_segmentation.analysis`
