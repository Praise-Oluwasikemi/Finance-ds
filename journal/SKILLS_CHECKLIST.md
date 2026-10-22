# Skills Checklist

Tick a skill **only when you can do it without AI writing the code** (docs and Google are fine).
Change `- [ ]` to `- [x]` and add the date. Look back at this list whenever your confidence dips.

## Python & pandas (TS 1)
- [ ] Write functions with default arguments and docstrings
- [ ] Loops, conditionals, list/dict comprehensions
- [ ] Read CSVs, filter with boolean masks, select with `.loc`
- [ ] `groupby` + `agg` with several statistics at once
- [ ] `pivot_table`, `merge`, `pd.cut` / `pd.qcut`
- [ ] Handle missing values and explain the choice
- [ ] Detect outliers with the IQR rule
- [ ] Work with dates: `to_datetime`, `.dt`, `to_period`
- [ ] Handle errors with `try/except`

## Visualisation & statistics (TS 2)
- [ ] Histogram, boxplot, bar and line charts with labelled axes
- [ ] Seaborn: heatmap, pairplot, boxplot by group
- [ ] Explain mean vs median for skewed data
- [ ] Correlation, and why it isn't causation
- [ ] State H₀/H₁, run a test, and interpret the p-value in plain English
- [ ] Chi-square, t-test / Mann-Whitney, two-proportion z-test
- [ ] Confidence interval for a mean and for a proportion
- [ ] Bootstrap a confidence interval
- [ ] Explain statistical vs practical significance

## SQL (TS 4)
- [ ] SELECT / WHERE / ORDER BY / LIMIT
- [ ] GROUP BY + HAVING
- [ ] INNER JOIN and LEFT JOIN, and when to use each
- [ ] Subqueries
- [ ] Run SQL from Python with sqlite3 + pandas

## Machine learning (TS 3, 5, 6, 7)
- [ ] Train/test split and explain why it matters
- [ ] Linear & logistic regression
- [ ] Decision tree & random forest, with feature importance
- [ ] Cross-validation
- [ ] Confusion matrix, precision, recall, F1, ROC-AUC
- [ ] Handle imbalanced classes (class weights, SMOTE)
- [ ] Build a scikit-learn `Pipeline` with preprocessing
- [ ] Tune with GridSearchCV / RandomizedSearchCV
- [ ] Explain the bias-variance trade-off
- [ ] Turn model output into a ₦ business decision

## Unsupervised & time series (TS 8, 9)
- [ ] K-Means with elbow + silhouette
- [ ] Hierarchical clustering
- [ ] PCA for visualisation
- [ ] Decompose a time series (trend, seasonality, residual)
- [ ] Lag/rolling features and an ARIMA forecast vs a naive baseline

## Deployment & explainability (TS 10, 11)
- [ ] Save/load a model with joblib
- [ ] Build and deploy a Streamlit app
- [ ] SHAP / LIME explanations
- [ ] Discuss fairness and write a model card

## Communication (all modules)
- [ ] Write an executive summary with numbers for a non-technical manager
- [ ] Explain a project in 2 minutes, out loud
- [ ] Write a README for a project
- [ ] Publish a LinkedIn post about a project
