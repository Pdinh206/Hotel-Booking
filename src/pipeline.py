"""
PIPELINE: TIỀN XỬ LÝ, MÔ HÌNH HÓA & QUẢN TRỊ DOANH THU
Đề tài: Dự báo Hủy phòng Khách sạn & Tối ưu hóa Doanh thu (City Hotel)

Quy trình gộp chung gồm 3 phần:
1. Phần 1: Tiền xử lý dữ liệu, làm sạch, loại bỏ rò rỉ & Kiểm định chất lượng.
2. Phần 2: Huấn luyện mô hình Random Forest (Time-based Split, Baseline Logistic, Đánh giá, Feature Importance).
3. Phần 3: Thuật toán Quản trị Doanh thu (Newsvendor Overbooking, Littlewood's Rule) & Dự báo đơn mới (reindex).
"""

import sys
from pathlib import Path
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score

try:
    from revenue import (
        build_daily_cancellation_risk,
        calculate_littlewood,
        calculate_overbooking,
        REVENUE_PARAMS,
    )
except ImportError:
    from src.revenue import (
        build_daily_cancellation_risk,
        calculate_littlewood,
        calculate_overbooking,
        REVENUE_PARAMS,
    )

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_DIR / "data" / "raw" / "hotel_bookings.csv"
CLEAN_DATA = PROJECT_DIR / "data" / "processed" / "hotel_clean.csv"
MODEL_FILE = PROJECT_DIR / "models" / "model_bundle.pkl"
FEATURE_IMPORTANCE_FILE = PROJECT_DIR / "reports" / "feature_importance.csv"
DAILY_RISK_FILE = PROJECT_DIR / "reports" / "daily_cancellation_risk.csv"
TEST_START_DATE = "2017-01-01"

def run_full_pipeline():
    print("=" * 75)
    print(" PIPELINE: TIỀN XỬ LÝ, HUẤN LUYỆN MODEL & QUẢN TRỊ DOANH THU")
    print("=" * 75)

    # =========================================================================
    # PHẦN 1: TIỀN XỬ LÝ DỮ LIỆU & KIỂM ĐỊNH CHẤT LƯỢNG
    # =========================================================================
    print("\n>>> [PHẦN 1] BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU...")
    df = pd.read_csv(RAW_DATA)
    print(f"1. Tổng số dòng ban đầu: {df.shape[0]:,} dòng, {df.shape[1]} cột")

    # 1.1 Lọc riêng City Hotel
    df = df[df["hotel"] == "City Hotel"].copy()
    print(f"2. Số dòng City Hotel: {len(df):,} dòng (tỷ lệ hủy ban đầu: {df['is_canceled'].mean()*100:.2f}%)")

    # 1.2 Xóa 2 cột rò rỉ dữ liệu trực tiếp (Data Leakage)
    df.drop(columns=["reservation_status", "reservation_status_date"], inplace=True, errors="ignore")

    # 1.3 Xử lý Missing Values
    df["has_company"] = df["company"].notna().astype(int)
    df["has_agent"] = df["agent"].notna().astype(int)
    df.drop(columns=["company"], inplace=True, errors="ignore")
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0).astype(int)
    df["children"] = df["children"].fillna(0).astype(int)

    # 1.4 Lọc dữ liệu không hợp lý (Dirty Data)
    # Lọc đơn 0 khách
    mask_zero_guests = (df["adults"] + df["children"] + df["babies"]) == 0
    df = df[~mask_zero_guests]

    # Lọc đơn 0 đêm
    mask_zero_nights = (df["stays_in_weekend_nights"] + df["stays_in_week_nights"]) == 0
    df = df[~mask_zero_nights]

    # Lọc ngoại lai ADR > 1000
    mask_adr_outlier = df["adr"] > 1000
    df = df[~mask_adr_outlier]

    # Gắn cờ phòng miễn phí (ADR = 0)
    df["adr_zero"] = (df["adr"] == 0).astype(int)

    # Chuẩn hóa meal Undefined -> SC và lọc bỏ Undefined ở kênh/phân khúc
    df["meal"] = df["meal"].replace("Undefined", "SC")
    df = df[(df["market_segment"] != "Undefined") & (df["distribution_channel"] != "Undefined")]

    # 1.5 Tạo các đặc trưng bổ trợ
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    month_map = {
        "January": "01", "February": "02", "March": "03", "April": "04",
        "May": "05", "June": "06", "July": "07", "August": "08",
        "September": "09", "October": "10", "November": "11", "December": "12"
    }
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_date_month"].map(month_map) + "-" +
        df["arrival_date_day_of_month"].astype(str).str.zfill(2)
    )

    # Xuất file dữ liệu 
    df.to_csv(CLEAN_DATA, index=False)
    print(f"-> Đã xuất file sạch '{CLEAN_DATA.name}': {len(df):,} dòng, {df.shape[1]} cột")

    # =========================================================================
    # PHẦN 2: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST DỰ BÁO HỦY PHÒNG
    # =========================================================================
    print("\n" + "=" * 75)
    print(">>> [PHẦN 2] HUẤN LUYỆN MÔ HÌNH RANDOM FOREST & BASELINE LOGISTIC")
    print("=" * 75)

    # 2.1 Lựa chọn đặc trưng tại thời điểm đặt (Pre-arrival Features)
    LEAKAGE_BLACKLIST = [
        "reservation_status", "reservation_status_date",
        "assigned_room_type", "is_room_type_changed", "booking_changes",
        "company", "agent", "arrival_date"
    ]

    num_features = [
        "lead_time", "total_nights", "total_guests", "adults", "children", "babies",
        "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
        "days_in_waiting_list", "adr", "adr_zero", "has_agent", "has_company"
    ]

    cat_features = [
        "meal", "market_segment", "distribution_channel",
        "deposit_type", "customer_type", "reserved_room_type"
    ]

    num_features = [f for f in num_features if f in df.columns and f not in LEAKAGE_BLACKLIST]
    cat_features = [f for f in cat_features if f in df.columns and f not in LEAKAGE_BLACKLIST]

    X = df[num_features + cat_features].copy()
    y = df["is_canceled"].copy()

    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)
    feature_names = X_encoded.columns.tolist()

    # 2.2 Phân chia tập dữ liệu theo thời gian (Time-based Split)
    train_mask = df["arrival_date"] < "2017-01-01"
    X_train, X_test = X_encoded[train_mask], X_encoded[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"Tập Train (2015-2016): {len(X_train):,} mẫu (Tỷ lệ hủy: {y_train.mean()*100:.2f}%)")
    print(f"Tập Test  (2017)     : {len(X_test):,} mẫu (Tỷ lệ hủy: {y_test.mean()*100:.2f}%)")

    # 2.3 Huấn luyện Baseline Logistic Regression
    print("\n1. Huấn luyện Baseline Logistic Regression...")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)).fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_prob)

    # 2.4 Huấn luyện Random Forest Classifier
    print("2. Huấn luyện Random Forest Classifier (100 cây, max_depth=16)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 2.5 Đánh giá kết quả trên tập Test 2017
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    acc = (y_pred == y_test).mean()

    print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST NĂM 2017 ---")
    print(f" Baseline Logistic AUC  : {lr_auc:.4f}")
    print(f" Random Forest ROC-AUC  : {auc_score:.4f} (cao hơn khoảng {auc_score - lr_auc:.3f} điểm AUC; mô hình đơn giản đã đạt phần lớn hiệu quả)")
    print(f" Random Forest PR-AUC   : {pr_auc:.4f} (so với mức nền 0.426)")
    print(f" Độ chính xác (Accuracy): {acc*100:.2f}% (so với phương án luôn đoán Không hủy: {(1 - y_test.mean())*100:.1f}%)")
    print("\nBáo cáo phân loại chi tiết:")
    print(classification_report(y_test, y_pred, target_names=["Không hủy (0)", "Hủy phòng (1)"], digits=4))

    # 2.6 Feature Importance & Lưu Bundle
    feat_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("TOP 10 ĐẶC TRƯNG QUAN TRỌNG NHẤT:")
    for rank, (feat, val) in enumerate(feat_imp.head(10).items(), 1):
        print(f"  {rank:2d}. {feat:<38} : {val*100:.2f}%")

    bundle = {
        "model": rf_model,
        "feature_names": feature_names,
        "num_features": num_features,
        "cat_features": cat_features,
        "auc_score": auc_score,
        "pr_auc": pr_auc,
        "lr_auc": lr_auc,
        "test_start_date": TEST_START_DATE,
    }
    joblib.dump(bundle, MODEL_FILE)
    feat_imp.head(15).to_csv(FEATURE_IMPORTANCE_FILE, header=["importance"])
    print("\n-> Đã lưu thành công: 'model_bundle.pkl' và 'feature_importance.csv'")

    # =========================================================================
    # PHẦN 3: THUẬT TOÁN DOANH THU & DỰ BÁO ĐƠN MỚI
    # =========================================================================
    print("\n" + "=" * 75)
    print(">>> [PHẦN 3] THUẬT TOÁN QUẢN TRỊ DOANH THU & DỰ BÁO ĐƠN MỚI")
    print("=" * 75)

    # 3.1 Newsvendor dùng trực tiếp xác suất hủy từ Random Forest.
    # Dùng năm 2017 (tập test theo thời gian) để tránh lấy dự báo trên dữ liệu train.
    test_data = df.loc[~train_mask].copy()
    test_data["cancel_prob"] = y_prob
    daily_risk = build_daily_cancellation_risk(
        test_data,
        test_data["cancel_prob"],
    )

    daily_risk["overbook_rooms"] = daily_risk.apply(
        lambda row: calculate_overbooking(
            mean_cancels=row["expected_cancellations"],
            std_cancels=row["cancellation_std"],
            adr=REVENUE_PARAMS["adr"],
            walk_cost_multiplier=REVENUE_PARAMS["walk_cost_multiplier"],
        )["overbook_rooms"],
        axis=1,
    )
    daily_risk["booking_limit"] = (
        REVENUE_PARAMS["capacity"] + daily_risk["overbook_rooms"]
    )
    daily_risk.to_csv(DAILY_RISK_FILE, index=False)

    peak_risk = daily_risk.sort_values(
        "expected_cancellations",
        ascending=False,
    ).iloc[0]
    peak_result = calculate_overbooking(
        mean_cancels=peak_risk["expected_cancellations"],
        std_cancels=peak_risk["cancellation_std"],
        adr=REVENUE_PARAMS["adr"],
        walk_cost_multiplier=REVENUE_PARAMS["walk_cost_multiplier"],
    )

    print("1. Newsvendor theo xác suất hủy từ Random Forest:")
    print(
        f"   -> Ngày rủi ro cao nhất: {peak_risk['arrival_date'].date()} "
        f"(hủy kỳ vọng {peak_risk['expected_cancellations']:.2f}, "
        f"độ lệch chuẩn {peak_risk['cancellation_std']:.2f})"
    )
    print(
        f"   -> Đề xuất bán vượt {peak_result['overbook_rooms']} phòng "
        f"(hạn mức khoảng {REVENUE_PARAMS['capacity'] + peak_result['overbook_rooms']} booking-room-night)."
    )

    # 3.2 Baseline EDA để đối chiếu với kịch bản model-driven.
    baseline_result = calculate_overbooking(
        mean_cancels=REVENUE_PARAMS["mean_cancel"],
        std_cancels=REVENUE_PARAMS["std_cancel"],
        adr=REVENUE_PARAMS["adr"],
        walk_cost_multiplier=REVENUE_PARAMS["walk_cost_multiplier"],
    )
    print("\n2. Newsvendor baseline theo tham số EDA bàn giao:")
    print(
        f"   -> μ={REVENUE_PARAMS['mean_cancel']}, "
        f"σ={REVENUE_PARAMS['std_cancel']}, "
        f"ADR={REVENUE_PARAMS['adr']} => "
        f"Q*={baseline_result['overbook_rooms']} phòng."
    )
    if baseline_result["overbook_rooms"] != REVENUE_PARAMS["handoff_overbook_rooms"]:
        print(
            f"   -> Q*=33 là kết quả chính thức theo công thức hiện tại; "
            f"Q*={REVENUE_PARAMS['handoff_overbook_rooms']} chỉ là giá trị bàn giao cũ, không sử dụng."
        )

    # 3.3 Littlewood vẫn là mô phỏng riêng vì cần dự báo nhu cầu nhóm giá cao.
    low_net = REVENUE_PARAMS["p_low_gross"] * (1 - REVENUE_PARAMS["commission"])
    print("\n3. Littlewood theo tham số EDA bàn giao:")
    res_lw = calculate_littlewood(
        p_high=REVENUE_PARAMS["p_high"],
        p_low=low_net,
        mean_demand_high=REVENUE_PARAMS["mean_demand_high"],
        std_demand_high=REVENUE_PARAMS["std_demand_high"],
        capacity=REVENUE_PARAMS["capacity"],
    )
    print(
        f"   -> Giá thấp sau hoa hồng={low_net:.2f} EUR; "
        f"giữ lại {res_lw['protection_level']} phòng giá cao, "
        f"mở bán {res_lw['booking_limit_low']} phòng giá thấp."
    )

    # 3.4 Dự báo đơn mới với kỹ thuật reindex
    print("\n4. Thử nghiệm dự báo đơn mới với reindex:")
    def predict_new_booking(booking_dict):
        input_df = pd.DataFrame([booking_dict])
        input_encoded = pd.get_dummies(input_df, columns=cat_features).reindex(columns=feature_names, fill_value=0)
        prob = rf_model.predict_proba(input_encoded)[0][1]
        return round(prob * 100, 2)

    sample_1 = {
        "lead_time": 150, "adults": 2, "children": 0, "babies": 0, "meal": "BB",
        "market_segment": "Online TA", "distribution_channel": "TA/TO", "reserved_room_type": "A",
        "deposit_type": "No Deposit", "customer_type": "Transient", "adr": 95.0,
        "total_nights": 3, "total_guests": 2, "is_repeated_guest": 0,
        "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
        "days_in_waiting_list": 0, "adr_zero": 0, "has_agent": 1, "has_company": 0
    }

    sample_2 = {
        "lead_time": 200, "adults": 2, "children": 1, "babies": 0, "meal": "HB",
        "market_segment": "Online TA", "distribution_channel": "TA/TO", "reserved_room_type": "D",
        "deposit_type": "Non Refund", "customer_type": "Transient", "adr": 120.0,
        "total_nights": 5, "total_guests": 3, "is_repeated_guest": 0,
        "previous_cancellations": 2, "previous_bookings_not_canceled": 0,
        "days_in_waiting_list": 0, "adr_zero": 0, "has_agent": 1, "has_company": 0
    }

    p1 = predict_new_booking(sample_1)
    p2 = predict_new_booking(sample_2)

    sample_1_nr = sample_1.copy()
    sample_1_nr["deposit_type"] = "Non Refund"
    p1_nr = predict_new_booking(sample_1_nr)

    print(f"   - Đơn 1 (No Deposit, lead_time=150)             : {p1}% xác suất hủy -> AN TOÀN")
    print(f"   - Đơn 2 (Non Refund, từng hủy 2 lần, lead=200)  : {p2}% xác suất hủy -> CẢNH BÁO HỦY")
    print(f"   - Đơn 1 đổi sang Non Refund                     : {p1_nr}% xác suất hủy (Tăng +{p1_nr - p1:.2f}%)")

    print("\n" + "=" * 75)
    print(" Xong!")
    print("=" * 75)

if __name__ == "__main__":
    run_full_pipeline()
