# Peer-to-Peer Lending Risk Analysis

Loan default risk analysis on peer-to-peer lending data, checking whether
the platform's own risk grading actually predicts real-world outcomes, then
building a leakage-free classifier using only information known at the
moment a loan is issued.

## What this does

Given loan-level data from a peer-to-peer lending platform (loan amount,
term, interest rate, grade, purpose, and eventual repayment outcome), this
project asks two questions: does the platform's A-through-G risk grade
actually track real default rates, and can default be predicted using only
what's known when the loan is approved, not its later payment history.

The pipeline:

1. **EDA.** Loan status distribution and loan volume over time.
2. **Building a clean, resolved outcome.** Of the four status categories
   present (COMPLETED, CURRENT, CHARGEOFF, DEFAULTED), only COMPLETED and
   CHARGEOFF loans are used to define the target. CURRENT loans are still
   ongoing and haven't reached a final outcome yet, including them would
   mislabel loans that might still default later. DEFAULTED is a separate,
   very rare category (419 of 943,695 rows, 0.04%) and is excluded for
   having too few examples to matter.
3. **Grade calibration.** Actual charge-off rate calculated per risk grade,
   to check whether the platform's own grading system holds up against real
   outcomes.
4. **Leakage-free feature set.** Only origination-time features are used
   for prediction: loan amount, term, borrower rate, installment, grade, and
   loan purpose. Everything that happens after the loan is issued
   (principal paid, interest paid, late fees, days past due, payment dates)
   is deliberately excluded, since those columns are consequences of the
   loan's performance, not causes, and using them would leak the answer.
5. **Two models.** Logistic Regression and Random Forest, both
   class-balanced, trained and validated on an 80/20 stratified split of
   the resolved loans.

## Results

**Grade calibration** is the standout finding: the platform's risk grade
predicts real charge-off rates almost perfectly, rising monotonically from
grade A to grade G.

| Grade | Actual charge-off rate |
|---|---|
| A | 6.3% |
| B | 13.5% |
| C | 22.6% |
| D | 30.8% |
| E | 39.8% |
| F | 45.9% |
| G | 50.8% |

**Classification performance**, using only origination-time features:

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.64 | 0.62 |
| Precision (charge-off) | 0.31 | 0.31 |
| Recall (charge-off) | 0.64 | **0.68** |
| ROC-AUC | 0.693 | **0.699** |

Both models land in a similar, modest range once the leaky post-origination
columns are removed, which is the honest, expected outcome: predicting
default from only what's known at approval time is a genuinely hard
problem, it's the real constraint every actual lender faces. The clean
separation the grade calibration chart shows comes from the platform's own
grading already baking in information a simple two-model comparison here
doesn't have access to (like proprietary credit bureau data).

## Tech stack

Python, `pandas`/`numpy` (data handling), `matplotlib`/`seaborn`
(visualisation), `scikit-learn` (`LogisticRegression`,
`RandomForestClassifier`, `StandardScaler`, `train_test_split`,
classification metrics).

## Data

Uses a peer-to-peer lending dataset (NPCI Online Peer-to-Peer Lending,
sourced from Lending Club) with 943,695 loan records across 18 columns.
The raw CSVs aren't included in this repo; place `train.csv`, `test.csv`,
and `sample_submission.csv` in the expected input path before running.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook online-peer-to-peer-lending.ipynb
```
