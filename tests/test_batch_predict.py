from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

sample_customer = {
    "city_tier": "Tier1",
    "age_group": "25-34",
    "acquisition_channel": "Organic",
    "loyalty_tier": "Gold",
    "preferred_category": "Skin Care",
    "marketing_consent": 1,
    "recency_days": 20,
    "frequency_180d": 5,
    "monetary_180d": 5000,
    "return_rate_180d": 0.1,
    "avg_discount_pct_180d": 10,
    "avg_rating_180d": 4.5,
    "category_diversity_180d": 3,
    "ticket_count_90d": 1,
    "negative_ticket_rate_90d": 0,
    "avg_resolution_hours_90d": 24,
    "days_since_signup": 400,
    "sessions_30d": 12,
    "product_views_30d": 30,
    "cart_adds_30d": 5,
    "wishlist_adds_30d": 2,
    "abandoned_carts_30d": 1,
    "email_opens_30d": 4,
    "campaign_clicks_30d": 2,
    "last_visit_days_ago": 2
}

def test_batch_predict():

    payload = {
        "customers": [
            sample_customer,
            sample_customer
        ]
    }

    response = client.post(
        "/batch_predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2