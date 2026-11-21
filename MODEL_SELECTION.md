# Model selection

This comparison uses the results supplied from the latest successful MLflow training batch.

## The question

The model predicts whether a customer will churn. A missed churn prediction may cost the business a retention opportunity, while a false churn prediction may lead to unnecessary offers or interventions for a customer who would have stayed.

## Candidates

| Candidate | Why I tried it | AUC | F1 | F1 per slice (lowest) | p95 latency (ms) | Size (KB) | MLflow run ID |
|---|---|---|---|---|---|---|---|
| Logistic Regression | A simple baseline that can provide useful predictions with a small model and low prediction latency. | 0.7982 | 0.4948 | 0.2500 (one-year and two-year contracts) | 6.5802 | 3.5479 | af7b547d66884cd8b23ea7b21667e638 |
| Random Forest | To test whether multiple decision trees can capture nonlinear patterns and interactions that the linear baseline misses. | 0.7589 | 0.4536 | 0.2222 (one-year contracts) | 20.1446 | 3160.127 | e643cb8476b34f28985d1cdd36a17146 |
| Gradient Boosting | To test whether trees trained in sequence can improve predictions by correcting earlier errors. | 0.7476 | 0.4167 | 0.2222 (two-year contracts) | 5.9785 | 141.6758 | e0140a884937487dbb16f17ddf897ca1 |

## The champion

I recommend Logistic Regression as the provisional champion. It has the highest overall F1 (0.4948) and AUC (0.7982), the smallest model size (3.5479 KB), and relatively low p95 prediction latency (6.5802 ms).

Gradient Boosting is slightly faster at 5.9785 ms, but I accept this latency disadvantage because Logistic Regression has better overall F1 and AUC and a smaller model. Random Forest has lower overall F1 and AUC, higher latency, and a larger model.

The choice still has a weakness: Logistic Regression's F1 is only 0.2500 on both one-year and two-year contract slices, compared with its overall F1 of 0.4948. These groups need further review. This is a provisional recommendation; no model promotion or deployment is claimed here.

## My promotion rules

- Latency budget (p95): **25 ms**. This is the Starter's initial value. All three candidates met this budget in our local measurements (6.58 ms, 20.14 ms, and 5.98 ms), and it provides a practical upper limit for prediction latency.
- Minimum gain to replace the champion: **0.005 absolute F1 points**. This is the Starter's initial value. It prevents replacing the current champion for extremely small F1 improvements. It is a provisional engineering rule, not a statistically validated threshold.

## What I would check next

One production risk is unreliable predictions for small or underperforming contract slices. I would check the test sample count and prediction errors for each contract type, especially the one-year and two-year groups. Small samples can make slice F1 unstable. I would also monitor changes in contract mix and customer behavior, since data drift could reduce overall and slice performance after deployment.
