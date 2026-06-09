# 2 Minute Video Script

Hi, my name is Rohitasva, and this is my submission for Task 3, "The Baseline
Beater."

The starter project gave us a marketing campaign dataset and a baseline model
that predicts the `Response` column. `Response` tells us whether a customer
accepted the campaign. The original notebook trained a basic Logistic Regression
model after dropping all non-numeric columns and filling missing values with
zero. When I reran that baseline in this environment, the F1-score was 0.1882,
which is low because the dataset is imbalanced. Only about 14.9% of customers
responded yes, so a model can get decent accuracy by mostly predicting no, but
that fails to find the important positive cases.

My main improvement was replacing the numeric-only baseline with a full
feature-engineered gradient boosting pipeline. I kept categorical columns like
`Education` and `Marital_Status` using one-hot encoding, handled missing numeric
values with the median, and created behavior-based features such as total spend,
total purchases, spend per purchase, age, customer tenure, number of children,
income per child, and how many previous campaigns the customer accepted.

The mathematical reason this helps is that F1-score balances precision and
recall using the formula 2 times precision times recall divided by precision
plus recall. Since the positive class is rare, the default 0.50 probability
threshold was too conservative and missed many true responders. I used a
validation split to choose the probability threshold that maximized F1, then
tested once on the holdout set. I also used HistGradientBoosting because boosted
trees can capture nonlinear patterns and feature interactions, for example how
high spending combined with low recency and previous campaign acceptance can
indicate a likely responder.

The final result is a test F1-score of 0.6076, compared with the rerun baseline
of 0.1882. That is a 222.8% relative improvement, which is well above the
required 20%. The code is reproducible: you can run `python baseline_beater.py`
and it prints the baseline score, selected threshold, final F1-score, accuracy,
precision, recall, and classification report.

So in summary, the biggest issue was not just the model type. It was that the
baseline ignored categorical and behavioral information and used a threshold
that was poorly matched to an imbalanced classification problem. My solution
fixes both of those issues in a clean, explainable pipeline.
