import joblib
import pandas as pd

model = joblib.load("models/model.pkl")


def get_risk_level(prob):

    if prob >= 0.7:
        return "high"

    elif prob >= 0.4:
        return "medium"

    return "low"


def generate_explanation(row):

    reasons = []

    if row["recency_days"] > 60:
        reasons.append("high recency")

    if row["sessions_30d"] < 3:
        reasons.append("low engagement")

    if row["ticket_count_90d"] > 2:
        reasons.append("many support issues")

    if not reasons:
        return "customer appears stable"

    return ", ".join(reasons)


def predict_customer(customer):

    df = pd.DataFrame([customer])

    # Engineered features used during training

    df["avg_order_value"] = (
        df["monetary_180d"] /
        df["frequency_180d"].replace(0, 1)
    )

    df["engagement_ratio"] = (
        df["sessions_30d"] /
        df["days_since_signup"].replace(0, 1)
    )

    df["email_click_rate"] = (
        df["campaign_clicks_30d"] /
        df["email_opens_30d"].replace(0, 1)
    )

    df["view_to_cart_ratio"] = (
        df["cart_adds_30d"] /
        df["product_views_30d"].replace(0, 1)
    )

    df["abandonment_rate"] = (
        df["abandoned_carts_30d"] /
        df["cart_adds_30d"].replace(0, 1)
    )

    df["tickets_per_order"] = (
        df["ticket_count_90d"] /
        df["frequency_180d"].replace(0, 1)
    )

    df["revenue_per_session"] = (
        df["monetary_180d"] /
        df["sessions_30d"].replace(0, 1)
    )

    df["tenure_years"] = (
        df["days_since_signup"] / 365
    )

    probability = float(
        model.predict_proba(df)[0][1]
    )

    prediction = int(
        model.predict(df)[0]
    )

    return {
        "churn_probability": round(probability, 4),
        "predicted_class": prediction,
        "risk_level": (
            "high" if probability >= 0.7
            else "medium" if probability >= 0.4
            else "low"
        ),
        "risk_explanation": "Prediction generated successfully."
    }