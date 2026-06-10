# D2C Customer Churn Prediction API

## ML Workflow

1. Load churn dataset
2. Preprocess features
3. Train Logistic Regression model
4. Evaluate using Accuracy, Precision, Recall, F1, and ROC-AUC
5. Save metrics to metrics.json
6. Save trained model to models/model.pkl
7. Load model in FastAPI for predictions

## Project Overview

This project provides a FastAPI-based machine learning service for predicting customer churn risk.

The API loads a trained churn prediction model and returns:

* Churn probability
* Predicted churn class
* Risk level (Low, Medium, High)
* Human-readable risk explanation

The service is designed to support CRM and retention teams by identifying customers who may be at risk of churning.

---

## Project Structure

```text
D2C_Part_4/
│
├── app/
│   ├── main.py
│   ├── predictor.py
│   ├── schemas.py
│   └── __init__.py
│
├── models/
│   └── model.pkl
│
├── tests/
│   └── test_api.py
│
├── monitoring_plan.md
├── responsible_use.md
├── requirements.txt
├── Dockerfile
├── README.md
└── train_model.py
```

---

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/Mansi-3s/Part_04_D2C-Churn-Prediction-API.git
cd Part_04_D2C-Churn-Prediction-API
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## Docker Setup

Build the Docker image:

```bash
docker build -t churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 --name churn-container churn-api
```

Verify the API is running:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

---

## API Endpoints

### GET /health

Health check endpoint.

Response:

```json
{
  "status": "ok"
}
```

---

### POST /predict

Predict churn risk for a single customer.

Sample Request:

```json
{
  "recency": 15,
  "frequency": 10,
  "monetary": 2500,
  "support_tickets": 2,
  "avg_discount_pct": 10
}
```

Sample Response:

```json
{
  "churn_probability": 0.72,
  "predicted_class": 1,
  "risk_level": "high",
  "risk_explanation": "Customer shows strong churn indicators such as low engagement and elevated support interactions."
}
```

---

### POST /batch_predict

Predict churn risk for multiple customers.

Sample Request:

```json
{
  "customers": [
    {
      "recency": 15,
      "frequency": 10,
      "monetary": 2500,
      "support_tickets": 2,
      "avg_discount_pct": 10
    },
    {
      "recency": 90,
      "frequency": 1,
      "monetary": 300,
      "support_tickets": 8,
      "avg_discount_pct": 40
    }
  ]
}
```

Sample Response:

```json
{
  "predictions": [
    {
      "churn_probability": 0.72,
      "predicted_class": 1,
      "risk_level": "high"
    },
    {
      "churn_probability": 0.15,
      "predicted_class": 0,
      "risk_level": "low"
    }
  ]
}
```

---

## Running Tests

Execute API tests:

```bash
pytest
```

Expected output:

```text
3 passed
```

---

## Model Notes

The churn prediction model was trained using the D2C customer dataset.

The model uses customer behavioral and engagement features to estimate churn risk.

Saved model artifact:

```text
models/model.pkl
```

If the model file is not present, it can be regenerated using:

```bash
python train_model.py
```

---

## Monitoring and Responsible Use

Additional project documentation:

* monitoring_plan.md
* responsible_use.md

These documents describe:

* Data drift monitoring
* Prediction distribution tracking
* Business outcome monitoring
* API error monitoring
* Model retraining triggers
* Responsible use guidelines for retention teams

```
```
