"""Các hàm dùng chung cho dự báo hủy và mô phỏng quản trị doanh thu.

Random Forest dự báo xác suất hủy ở cấp booking. Các hàm bên dưới tổng hợp
các xác suất đó theo ngày lưu trú để làm đầu vào cho Newsvendor. Littlewood
được giữ độc lập vì nó còn cần một dự báo nhu cầu riêng cho nhóm giá cao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# Bộ tham số thực nghiệm được Bạn 1 bàn giao.
# p_low_gross là giá OTA/TA trước hoa hồng; app sẽ tính giá thực nhận.
REVENUE_PARAMS = {
    # 250 là công suất kịch bản; peak occupancy quan sát được từ EDA là 226.
    "capacity": 250,
    "mean_cancel": 41.7,
    "std_cancel": 35.2,
    # ADR tính trên hotel_clean.csv; 106.02 là giá trị được bàn giao trước đó.
    "adr": 105.79,
    "adr_handoff_reported": 106.02,
    "walk_cost_multiplier": 1.5,
    "commission": 0.20,
    "p_high": 115.0,
    "p_low_gross": 90.0,
    "mean_demand_high": 25.0,
    "std_demand_high": 8.0,
    # Giá trị lịch sử trong biên bản bàn giao; chỉ dùng để audit, không ghi đè công thức.
    "handoff_overbook_rooms": 28,
}


def score_cancellation_probability(
    data: pd.DataFrame,
    bundle: dict,
) -> pd.Series:
    """Tính P(hủy) cho từng booking bằng model trong bundle."""

    feature_names = bundle["feature_names"]
    num_features = bundle["num_features"]
    cat_features = bundle["cat_features"]

    feature_columns = num_features + cat_features
    missing = [column for column in feature_columns if column not in data.columns]
    if missing:
        raise KeyError(f"Thiếu cột đầu vào cho model: {missing}")

    input_data = data[feature_columns].copy()
    encoded = pd.get_dummies(input_data, columns=cat_features)
    encoded = encoded.reindex(columns=feature_names, fill_value=0)

    model = bundle["model"]
    cancel_class = list(model.classes_).index(1)
    probabilities = model.predict_proba(encoded)[:, cancel_class]

    return pd.Series(probabilities, index=data.index, name="cancel_prob")


def aggregate_cancellation_risk(probabilities) -> tuple[float, float]:
    """Tổng hợp P(hủy) thành kỳ vọng và độ lệch chuẩn số hủy.

    Giả định các booking độc lập có phương sai Bernoulli p(1-p).
    """

    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 0.0, 1.0)

    mean_cancel = float(probabilities.sum())
    std_cancel = float(
        np.sqrt(np.sum(probabilities * (1.0 - probabilities)))
    )
    return mean_cancel, std_cancel


def build_daily_cancellation_risk(
    data: pd.DataFrame,
    probabilities,
) -> pd.DataFrame:
    """Tổng hợp rủi ro hủy theo từng ngày phòng được sử dụng.

    Mỗi booking được xem là dùng một phòng. Booking nhiều đêm đóng góp rủi ro
    vào từng ngày trong thời gian lưu trú; đây là đơn vị phù hợp hơn so với
    cộng đơn thuần theo ngày nhận phòng.
    """

    if "arrival_date" not in data.columns or "total_nights" not in data.columns:
        raise KeyError("Cần có arrival_date và total_nights để tổng hợp theo ngày")

    working = data[["arrival_date", "total_nights"]].copy()
    working["cancel_prob"] = np.asarray(probabilities, dtype=float)
    working["arrival_date"] = pd.to_datetime(working["arrival_date"])

    records = []
    for row in working.itertuples(index=False):
        if pd.isna(row.arrival_date):
            continue

        nights = max(1, int(row.total_nights))
        probability = float(np.clip(row.cancel_prob, 0.0, 1.0))
        variance = probability * (1.0 - probability)

        for offset in range(nights):
            records.append(
                {
                    "arrival_date": row.arrival_date + pd.Timedelta(days=offset),
                    "bookings": 1,
                    "expected_cancellations": probability,
                    "cancellation_variance": variance,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "arrival_date",
                "bookings",
                "expected_cancellations",
                "cancellation_std",
            ]
        )

    daily = (
        pd.DataFrame(records)
        .groupby("arrival_date", as_index=False)
        .agg(
            bookings=("bookings", "sum"),
            expected_cancellations=("expected_cancellations", "sum"),
            cancellation_variance=("cancellation_variance", "sum"),
        )
    )
    daily["cancellation_std"] = np.sqrt(daily["cancellation_variance"])
    return daily.drop(columns="cancellation_variance")


def calculate_overbooking(
    mean_cancels: float,
    std_cancels: float,
    adr: float,
    walk_cost_multiplier: float = 1.5,
) -> dict:
    """Tính buffer overbooking theo xấp xỉ Newsvendor."""

    underage_cost = float(adr)
    overage_cost = float(walk_cost_multiplier * adr)
    if underage_cost <= 0 or overage_cost <= 0:
        raise ValueError("ADR và chi phí chuyển khách phải lớn hơn 0")

    critical_ratio = underage_cost / (underage_cost + overage_cost)
    if std_cancels <= 0:
        overbook_rooms = int(round(mean_cancels))
    else:
        z_score = float(norm.ppf(critical_ratio))
        overbook_rooms = max(
            0,
            int(round(mean_cancels + z_score * std_cancels)),
        )

    return {
        "critical_ratio": round(critical_ratio, 4),
        "overbook_rooms": overbook_rooms,
    }


def calculate_littlewood(
    p_high: float,
    p_low: float,
    mean_demand_high: float,
    std_demand_high: float,
    capacity: int,
) -> dict:
    """Tính mức bảo vệ phòng cho nhóm giá cao theo Littlewood."""

    if p_high <= p_low:
        return {
            "protection_level": 0,
            "booking_limit_low": int(capacity),
        }

    target_probability = 1.0 - (p_low / p_high)
    if std_demand_high <= 0:
        protection_level = int(round(mean_demand_high))
    else:
        z_score = norm.ppf(target_probability)
        protection_level = int(round(mean_demand_high + z_score * std_demand_high))

    protection_level = max(0, min(int(capacity), protection_level))
    return {
        "protection_level": protection_level,
        "booking_limit_low": max(0, int(capacity) - protection_level),
    }
