# Monitoring Plan

## 1. Data Drift

Monitor incoming customer features and compare them with training data.

Examples:

- Recency distribution
- Frequency distribution
- Monetary value distribution
- Engagement metrics

If major shifts occur, investigate model performance.

---

## 2. Prediction Distribution

Track:

- Percentage predicted as churn
- Average churn probability
- High-risk customer count

Unexpected changes may indicate drift.

---

## 3. Business Outcomes

Track:

- Actual churn rate
- Retention campaign success rate
- Revenue retained
- Customer lifetime value

Compare model predictions with actual outcomes.

---

## 4. API Monitoring

Track:

- Request volume
- Response time
- Error rate
- Failed predictions

Set alerts for unusual spikes.

---

## 5. Retraining Triggers

Retrain model when:

- Prediction quality drops
- Data drift exceeds threshold
- New customer behaviour emerges
- Every 3-6 months as routine maintenance