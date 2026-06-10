from fastapi import FastAPI

from app.schemas import (
    CustomerInput,
    BatchPredictRequest
)

from app.predictor import predict_customer

app = FastAPI(
    title="Customer Churn API"
)

@app.get("/")
def root():
    return {
        "message": "Customer Churn API is running"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: CustomerInput):

    return predict_customer(
        customer.model_dump()
    )


@app.post("/batch_predict")
def batch_predict(
    request: BatchPredictRequest
):

    results = []

    for customer in request.customers:

        results.append(
            predict_customer(
                customer.model_dump()
            )
        )

    return results