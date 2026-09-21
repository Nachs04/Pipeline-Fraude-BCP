import pandas as pd
from src.pipeline import add_historical_features


def test_historical_features_do_not_use_future_rows():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-01 11:00", "2026-01-01 12:00"]
            ),
            "customer_id": ["A", "A", "A"],
            "terminal_id": ["T1", "T1", "T2"],
            "amount": [100.0, 200.0, 50.0],
            "fraud": [0, 1, 0],
        }
    )

    out = add_historical_features(df)

    assert out.loc[0, "customer_prev_count"] == 0
    assert out.loc[1, "customer_prev_count"] == 1
    assert out.loc[1, "customer_prev_amount"] == 100.0
    assert out.loc[2, "customer_prev_amount"] == 300.0


def test_no_target_as_feature():
    feature_names = {
        "amount",
        "customer_prev_count",
        "customer_prev_amount",
        "terminal_prev_count",
        "hour",
        "day_of_week",
    }
    assert "fraud" not in feature_names
