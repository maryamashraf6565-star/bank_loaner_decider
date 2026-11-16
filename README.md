# Bank Loan Approval Classifier

A single-neuron (logistic regression) classifier implemented **from scratch in NumPy** — sigmoid, forward pass, and gradient descent all hand-written, no Scikit-learn model used.

## Why this project
I wanted to prove I understand what happens *inside* `sklearn.linear_model.LogisticRegression`, not just how to call it.

## Dataset
A small illustrative dataset (6 rows) with three applicant features — age, salary (monthly), debt (in units of 10,000) — plus a Yes/No approval label. The final row is a hypothetical new applicant ("Me") with no label, used for prediction.

| Customer | Age | Salary | Debt | Approved |
|---|---|---|---|---|
| Customer 1 | 16 | 0 | 0 | No |
| Customer 2 | 24 | 5000 | 0 | Yes |
| Customer 3 | 23 | 6000 | 10 | Yes |
| Customer 4 | 28 | 5000 | 50 | No |
| Customer 5 | 40 | 15000 | 200 | Yes |
| **Me** | **50** | **12000** | **30** | **?** |

## How it works
1. **Split:** first 5 rows = training, last row ("Me") = prediction target
2. **Standardization:** features scaled with `StandardScaler`, fit on train only and applied to the "Me" row with the same rules
3. **Labels:** Yes/No mapped to 1/0
4. **Model:** single neuron — `z = Xw + b`, `a = sigmoid(z)`
5. **Training:** gradient descent, **100 epochs, learning rate 0.01**, weights updated as `w -= lr * dw`, `b -= lr * db`
6. **Numerical stability:** sigmoid input clipped to ±500 to prevent `exp` overflow

## Result
Prediction for the "Me" row: **Loan Approved, probability 0.76**.

Final learned weights: `[0.480, -0.116, 0.906]`, bias `≈ 0.0`.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook bank_loaning_decider.ipynb .
```
## What I learned
What gradient descent actually updates epoch by epoch

Why standardizing features (and reusing the scaler on new data) matters

Why naive sigmoid implementations overflow, and how clipping fixes it
