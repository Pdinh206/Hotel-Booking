"""Dashboard City Hotel. Chạy: python -m streamlit run app.py."""

from pathlib import Path
import sys
import warnings

# Streamlit Cloud chạy entrypoint từ thư mục app/. Thêm thư mục gốc của
# repository vào sys.path để import src.revenue ổn định cả local lẫn Cloud.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency, ttest_ind

from src.revenue import (
    build_daily_cancellation_risk,
    calculate_littlewood,
    calculate_overbooking,
    REVENUE_PARAMS,
    score_cancellation_probability,
)


# ==================================================
# 1. CẤU HÌNH VÀ TRANG TRÍ GIAO DIỆN
# ==================================================

st.set_page_config(
    page_title="City Hotel | Quản trị doanh thu",
    page_icon="🏨",
    layout="wide"
)

BLUE = "#005BAA"
SKY = "#69A9DE"
INK = "#183655"

st.html("""
<style>
    .block-container {
        max-width: 1500px;
        padding: 2.4rem 2.6rem 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #DFE8F2;
    }

    h1, h2, h3 {
        letter-spacing: -0.6px;
    }

    h2 {
        font-size: 1.4rem !important;
        color: #183655;
    }

    h3 {
        font-size: 1.08rem !important;
        color: #183655;
    }

    /* Thẻ chỉ số */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DFE8F2;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 120px;
        box-shadow: 0 3px 12px rgba(17, 58, 99, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: #5E7186;
        font-size: 13px;
    }

    [data-testid="stMetricValue"] {
        color: #005BAA;
        font-weight: 700;
        font-size: clamp(20px, 1.75vw, 30px);
    }

    /* Khung nội dung */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 14px;
    }

    [data-testid="stForm"] {
        border-radius: 14px;
        border-color: #DFE8F2;
        padding: 22px;
    }

    /* Nút dự báo */
    [data-testid="stFormSubmitButton"] button {
        background: #005BAA;
        color: white;
        border: 1px solid #005BAA;
        border-radius: 9px;
        padding: 9px 22px;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background: #004785;
        color: white;
    }

    /* Nhận diện trên thanh bên */
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 6px 0 24px;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: #005BAA;
        color: white;
        font-size: 17px;
        font-weight: 800;
    }

    .brand-name {
        font-size: 19px;
        font-weight: 750;
        color: #183655;
    }

    .brand-sub {
        font-size: 12px;
        color: #6C7D90;
        margin-top: 1px;
    }

    /* Tiêu đề đầu trang */
    .hero {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
        padding: 0 0 25px;
        margin-bottom: 21px;
        border-bottom: 1px solid #DFE8F2;
    }

    .eyebrow {
        color: #005BAA;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 1.8px;
        margin-bottom: 9px;
    }

    .hero h1 {
        color: #183655;
        font-size: clamp(25px, 2.6vw, 35px);
        font-weight: 750;
        line-height: 1.25;
        margin: 0;
        padding: 0;
    }

    .hero p {
        color: #687C91;
        font-size: 14px;
        margin: 10px 0 0;
    }

    .period {
        flex-shrink: 0;
        border: 1px solid #DFE8F2;
        border-radius: 10px;
        padding: 12px 17px;
        background: #F7FAFE;
        color: #183655;
        font-size: 13px;
    }

    .period span {
        display: block;
        color: #738499;
        font-size: 11px;
        margin-bottom: 3px;
    }

    .result-note {
        background: #EDF5FF;
        border-left: 3px solid #005BAA;
        padding: 14px 18px;
        border-radius: 0 9px 9px 0;
        color: #183655;
        font-size: 14px;
    }

    .sidebar-note {
        font-size: 12px;
        line-height: 1.65;
        color: #687C91;
        margin-top: 12px;
    }

    .footer {
        border-top: 1px solid #DFE8F2;
        margin-top: 30px;
        padding-top: 15px;
        color: #7A899A;
        font-size: 12px;
    }

    /* Hiệu ứng xuất hiện nhẹ */
    @keyframes hotel-enter {
        from {
            opacity: 0;
            transform: translateY(7px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .hero {
        animation: hotel-enter 300ms ease-out backwards;
    }

    [data-testid="stMetric"], .result-note {
        animation: hotel-enter 250ms ease-out backwards;
    }

    /* Chuyển động khi tương tác */
    [data-testid="stMetric"] {
        transition:
            transform 180ms ease-out,
            box-shadow 180ms ease-out,
            border-color 180ms ease-out;
    }

    [data-testid="stFormSubmitButton"] button {
        transition:
            background-color 160ms ease-out,
            box-shadow 160ms ease-out,
            transform 160ms ease-out;
    }

    [data-testid="stFormSubmitButton"] button:active {
        transform: translateY(1px);
    }

    [data-testid="stTab"] {
        transition:
            color 160ms ease-out,
            background-color 160ms ease-out;
        border-radius: 8px 8px 0 0;
    }

    [data-testid="stTab"][aria-selected="true"] {
        background-color: #EDF5FF;
    }

    @media (hover: hover) and (pointer: fine) {
        [data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: #9DC2E5;
            box-shadow: 0 9px 22px rgba(0, 75, 145, 0.09);
        }

        [data-testid="stFormSubmitButton"] button:hover {
            box-shadow: 0 4px 12px rgba(0, 75, 145, 0.16);
        }
    }

    /* Tắt hiệu ứng nếu máy bật chế độ giảm chuyển động */
    @media (prefers-reduced-motion: reduce) {
        .hero,
        .result-note,
        [data-testid="stMetric"],
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stTab"] {
            animation: none !important;
            transition: none !important;
            transform: none !important;
        }
    }

    /* Điều chỉnh cho màn hình nhỏ */
    @media (max-width: 800px) {
        .block-container {
            padding: 1.6rem 1rem 2rem;
        }

        .hero {
            align-items: flex-start;
            flex-direction: column;
            gap: 14px;
        }

        .period {
            padding: 8px 12px;
        }
    }
</style>
""")


# ==================================================
# 2. ĐỌC DỮ LIỆU VÀ MÔ HÌNH
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        PROJECT_DIR / "data" / "processed" / "hotel_clean.csv",
        parse_dates=["arrival_date"]
    )


@st.cache_data
def load_eda():
    return pd.read_excel(
        PROJECT_DIR / "reports" / "bang_du_lieu_va_chi_so_eda.xlsx",
        sheet_name=["channel_summary", "lead_stats"]
    )


@st.cache_resource
def load_model():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        bundle = joblib.load(PROJECT_DIR / "models" / "model_bundle.pkl")

    version_warning = any(
        "unpickle estimator" in str(w.message)
        for w in captured
    )

    return bundle, version_warning


@st.cache_data
def load_daily_model_risk(data):
    """Tạo rủi ro hủy theo ngày từ tập dữ liệu ngoài giai đoạn train."""

    bundle, _ = load_model()
    test_start = pd.Timestamp(
        bundle.get("test_start_date", "2017-01-01")
    )
    candidates = data[data["arrival_date"] >= test_start].copy()

    probabilities = score_cancellation_probability(candidates, bundle)
    return build_daily_cancellation_risk(candidates, probabilities)


@st.cache_data
def make_daily_data(data):
    # Phân bổ từng đêm lưu trú để đơn qua tháng
    # được tính đúng vào từng tháng.
    records = []

    stays = data[data["is_canceled"] == 0]

    for row in stays.itertuples(index=False):
        for offset in range(int(row.total_nights)):
            day = row.arrival_date + pd.Timedelta(days=offset)
            records.append((day, row.adr))

    nights = pd.DataFrame(
        records,
        columns=["date", "revenue"]
    )

    daily = nights.groupby("date").agg(
        occupied_rooms=("revenue", "size"),
        revenue=("revenue", "sum")
    )

    calendar = pd.date_range(
        data["arrival_date"].min(),
        data["arrival_date"].max()
    )

    return daily.reindex(
        calendar,
        fill_value=0
    ).rename_axis("date")


# ==================================================
# 3. CÁC HÀM HIỂN THỊ BIỂU ĐỒ
# ==================================================

def chart_style(chart):
    return (
        chart
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#61758B",
            titleColor="#61758B",
            labelFontSize=11,
            titleFontSize=11,
            gridColor="#EAF0F7",
            domain=False,
            tickSize=0,
            labelPadding=8,
            titlePadding=12
        )
        .configure_legend(
            labelColor="#52657B",
            title=None,
            orient="bottom"
        )
    )


def monthly_line(data, field, label, color):
    chart = (
        alt.Chart(data)
        .mark_line(
            color=color,
            strokeWidth=2.5,
            point=True
        )
        .encode(
            x=alt.X(
                "month:T",
                title=None,
                axis=alt.Axis(
                    format="%m/%y",
                    tickCount=7
                )
            ),
            y=alt.Y(
                f"{field}:Q",
                title=label,
                scale=alt.Scale(zero=True)
            ),
            tooltip=[
                alt.Tooltip(
                    "month:T",
                    title="Tháng",
                    format="%m/%Y"
                ),
                alt.Tooltip(
                    f"{field}:Q",
                    title=label,
                    format=",.2f"
                )
            ]
        )
        .properties(height=250)
    )

    st.altair_chart(
        chart_style(chart),
        width="stretch"
    )


def p_value_text(value):
    # Không hiển thị p = 0 do giới hạn độ chính xác số thực.
    if value < 0.001:
        return "< 0,001"

    return f"= {value:.4f}"


try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "Thiếu hotel_clean.csv. "
        "Hãy đặt file này cùng thư mục với app.py."
    )
    st.stop()


# ==================================================
# 4. THANH ĐIỀU HƯỚNG
# ==================================================

with st.sidebar:
    st.html("""
    <div class="brand">
        <div class="brand-mark">CH</div>
        <div>
            <div class="brand-name">City Hotel</div>
            <div class="brand-sub">Quản trị doanh thu</div>
        </div>
    </div>
    """)

    page = st.radio(
        "ĐIỀU HƯỚNG",
        [
            "1. Tổng quan",
            "2. Phân tích dữ liệu",
            "3. Dự báo hủy",
            "4. What-if"
        ]
    )

    st.divider()

    st.markdown("**Tham số kịch bản**")

    capacity = st.slider(
        "Công suất giả định (phòng)",
        100, 400, REVENUE_PARAMS["capacity"], 10
    )

    commission = st.slider(
        "Hoa hồng TA/TO giả định (%)",
        10, 30, int(REVENUE_PARAMS["commission"] * 100)
    ) / 100

    st.html("""
    <div class="sidebar-note">
        Công suất mặc định 250 phòng là giả định mô phỏng.
        Các tham số dùng cho chỉ số công suất
        và phần mô phỏng What-if.
    </div>
    """)

    st.divider()

    st.caption(
        "Nguồn: Hotel Booking Demand\n\n"
        "Phạm vi: City Hotel · Đơn vị: EUR"
    )


# ==================================================
# 5. TIÊU ĐỀ TỪNG TRANG
# ==================================================

page_titles = {
    "1. Tổng quan": (
        "Tổng quan hoạt động",
        "Theo dõi đặt phòng, tình trạng hủy "
        "và hiệu quả sử dụng phòng."
    ),
    "2. Phân tích dữ liệu": (
        "Phân tích đặt phòng",
        "Khám phá kênh bán, thời gian đặt trước "
        "và kết quả kiểm định."
    ),
    "3. Dự báo hủy": (
        "Dự báo xác suất hủy",
        "Nhập thông tin đặt phòng để sử dụng "
        "mô hình Random Forest."
    ),
    "4. What-if": (
        "Mô phỏng chính sách doanh thu",
        "Thử các giả định hoa hồng, bán vượt phòng "
        "và bảo vệ phòng."
    )
}

title, subtitle = page_titles[page]

st.html(f"""
<div class="hero">
    <div>
        <div class="eyebrow">
            CITY HOTEL / QUẢN TRỊ DOANH THU
        </div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    <div class="period">
        <span>DỮ LIỆU LỊCH SỬ</span>
        07/2015 – 08/2017
    </div>
</div>
""")


# ==================================================
# 6. TRANG TỔNG QUAN
# ==================================================

if page == "1. Tổng quan":

    left, right = st.columns([3, 1])

    with left:
        st.subheader("Các chỉ số chính")
        st.caption(
            "Các thẻ bên dưới được lọc theo năm khách đến."
        )

    with right:
        year = st.selectbox(
            "Năm đến",
            ["Tất cả"]
            + sorted(
                df["arrival_date_year"].unique().tolist()
            )
        )

    if year == "Tất cả":
        data = df
    else:
        data = df[df["arrival_date_year"] == year]

    stays = data[data["is_canceled"] == 0]

    room_nights = stays["total_nights"].sum()

    revenue = (
        stays["adr"] * stays["total_nights"]
    ).sum()

    weighted_adr = (
        revenue / room_nights
        if room_nights > 0
        else None
    )

    cards = st.columns(4)

    cards[0].metric(
        "Lượt đặt phòng",
        f"{len(data):,}"
    )

    cards[1].metric(
        "Tỷ lệ hủy",
        f"{data['is_canceled'].mean():.2%}"
    )

    cards[2].metric(
        "ADR đơn không hủy",
        f"{weighted_adr:.2f} EUR"
        if weighted_adr is not None
        else "—"
    )

    cards[3].metric(
        "Đêm phòng không hủy",
        f"{int(room_nights):,}"
    )

    st.caption(
        "ADR tính theo số đêm lưu trú của các đơn không hủy."
    )

    st.write("")

    # Biểu đồ tỷ lệ hủy theo kênh bán
    with st.container(border=True):
        st.subheader("Tỷ lệ hủy theo kênh bán")

        st.caption(
            "So sánh tỷ lệ đơn bị hủy trong từng kênh."
        )

        rates = (
            data.groupby("distribution_channel")["is_canceled"]
            .mean()
            .reset_index()
        )

        rates["rate"] = rates["is_canceled"] * 100

        bars = (
            alt.Chart(rates)
            .mark_bar(
                color=BLUE,
                cornerRadiusEnd=5,
                size=24
            )
            .encode(
                x=alt.X(
                    "rate:Q",
                    title="Tỷ lệ hủy (%)",
                    scale=alt.Scale(domain=[0, 100])
                ),
                y=alt.Y(
                    "distribution_channel:N",
                    title=None,
                    sort="-x"
                ),
                tooltip=[
                    alt.Tooltip(
                        "distribution_channel:N",
                        title="Kênh bán"
                    ),
                    alt.Tooltip(
                        "rate:Q",
                        title="Tỷ lệ hủy (%)",
                        format=".2f"
                    )
                ]
            )
        )

        labels = (
            bars
            .mark_text(
                align="left",
                dx=8,
                color=INK
            )
            .encode(
                text=alt.Text("rate:Q", format=".2f")
            )
        )

        st.altair_chart(
            chart_style(
                (bars + labels).properties(height=175)
            ),
            width="stretch"
        )

    # Tính chỉ số từng tháng lưu trú
    daily = make_daily_data(df)

    if year != "Tất cả":
        daily = daily[daily.index.year == year]

    monthly = daily.groupby(
        daily.index.to_period("M")
    ).agg(
        occupied_nights=("occupied_rooms", "sum"),
        revenue=("revenue", "sum"),
        observed_days=("occupied_rooms", "size")
    )

    available = capacity * monthly["observed_days"]

    monthly["occupancy"] = (
        monthly["occupied_nights"] / available * 100
    )

    monthly["revpar"] = (
        monthly["revenue"] / available
    )

    monthly["month"] = monthly.index.to_timestamp()

    plot_data = monthly.reset_index(drop=True)

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.subheader("Tỷ lệ lấp đầy phòng")

            st.caption(
                f"Theo tháng lưu trú · "
                f"Công suất giả định {capacity} phòng"
            )

            monthly_line(
                plot_data,
                "occupancy",
                "Occupancy (%)",
                BLUE
            )

    with c2:
        with st.container(border=True):
            st.subheader("Doanh thu trên phòng sẵn có")

            st.caption(
                "Theo tháng lưu trú · RevPAR"
            )

            monthly_line(
                plot_data,
                "revpar",
                "RevPAR (EUR)",
                "#3987C6"
            )

    if monthly["occupancy"].max() > 100:
        st.warning(
            "Occupancy vượt 100%: công suất giả định "
            "thấp hơn số phòng ghi nhận."
        )

    st.caption(
        "Giả định một đơn là một phòng. "
        "Khung báo cáo theo ngày đến của dữ liệu; "
        "không bao gồm khách đã đến trước 07/2015. "
        "Chỉ số tháng lưu trú có thể gồm đơn đến từ năm trước."
    )

    with st.expander("Bảng theo tháng và cách tính"):
        table = monthly.drop(columns="month").copy()
        table.index = table.index.astype(str)

        table.columns = [
            "Đêm phòng",
            "Doanh thu (EUR)",
            "Số ngày",
            "Occupancy (%)",
            "RevPAR (EUR)"
        ]

        st.dataframe(
            table.round(2),
            width="stretch"
        )

        st.caption(
            "Occupancy = Đêm phòng / (Công suất × Số ngày). "
            "RevPAR = Doanh thu phòng / (Công suất × Số ngày)."
        )

    with st.expander("Xem dữ liệu đặt phòng"):
        st.dataframe(
            data.head(100),
            hide_index=True,
            width="stretch"
        )


# ==================================================
# 7. TRANG PHÂN TÍCH DỮ LIỆU
# ==================================================

elif page == "2. Phân tích dữ liệu":

    st.caption(
        "Phân tích trên toàn bộ dữ liệu sạch; "
        "các thanh trượt kịch bản không thay đổi "
        "kết quả ở trang này."
    )

    try:
        tables = load_eda()

    except (FileNotFoundError, ValueError) as error:
        st.error(f"Không đọc được bảng EDA: {error}")

        st.info(
            "Cần file bang_du_lieu_va_chi_so_eda.xlsx "
            "với hai sheet channel_summary và lead_stats."
        )

        st.stop()

    with st.container(border=True):
        st.subheader("Hiệu quả theo kênh bán")

        names = {
            "distribution_channel": "Kênh bán",
            "total_bookings": "Lượt đặt",
            "canceled_count": "Đơn hủy",
            "total_nights": "Đêm đặt",
            "mean_adr": "ADR TB (EUR)",
            "median_lead_time": "Đặt trước trung vị (ngày)",
            "cancel_rate_%": "Tỷ lệ hủy (%)",
            "nights_share_%": "Tỷ trọng đêm đặt (%)"
        }

        st.dataframe(
            tables["channel_summary"]
            .rename(columns=names)
            .round(2),
            hide_index=True,
            width="stretch"
        )

        st.caption(
            "Đêm đặt và tỷ trọng gồm cả đơn hủy; "
            "ADR ở bảng này là trung bình theo đơn."
        )

    lead = tables["lead_stats"].copy()

    lead["lead_time_group"] = (
        lead["lead_time_group"].replace(
            {
                "< 14 ngày": "0–14 ngày",
                "14 - 30 ngày": "15–30 ngày"
            }
        )
    )

    with st.container(border=True):
        st.subheader("Thời gian đặt trước và tỷ lệ hủy")

        st.caption(
            "Các nhóm được sắp theo số ngày đặt trước tăng dần."
        )

        chart = (
            alt.Chart(lead)
            .mark_bar(
                color=BLUE,
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5
            )
            .encode(
                x=alt.X(
                    "lead_time_group:N",
                    title=None,
                    sort=lead["lead_time_group"].tolist(),
                    axis=alt.Axis(labelAngle=0)
                ),
                y=alt.Y(
                    "cancel_rate_%:Q",
                    title="Tỷ lệ hủy (%)",
                    scale=alt.Scale(domain=[0, 100])
                ),
                tooltip=[
                    alt.Tooltip(
                        "lead_time_group:N",
                        title="Đặt trước"
                    ),
                    alt.Tooltip(
                        "cancel_rate_%:Q",
                        title="Tỷ lệ hủy (%)",
                        format=".2f"
                    ),
                    alt.Tooltip(
                        "total:Q",
                        title="Số đơn",
                        format=","
                    )
                ]
            )
            .properties(height=260)
        )

        st.altair_chart(
            chart_style(chart),
            width="stretch"
        )

        with st.expander("Xem số liệu từng nhóm"):
            st.dataframe(
                lead.rename(
                    columns={
                        "lead_time_group": "Đặt trước",
                        "total": "Lượt đặt",
                        "canceled": "Đơn hủy",
                        "cancel_rate_%": "Tỷ lệ hủy (%)"
                    }
                ).round(2),
                hide_index=True
            )

    # Kiểm định thống kê
    contingency = pd.crosstab(
        df["distribution_channel"],
        df["is_canceled"]
    )

    chi2, p_chi, dof, expected = chi2_contingency(
        contingency
    )

    canceled = df.loc[
        df["is_canceled"] == 1,
        "adr"
    ].dropna()

    stayed = df.loc[
        df["is_canceled"] == 0,
        "adr"
    ].dropna()

    t_stat, p_t = ttest_ind(
        canceled,
        stayed,
        equal_var=False
    )

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.subheader("Kênh bán và tình trạng hủy")
            st.caption("KIỂM ĐỊNH CHI-SQUARE")

            st.write(
                f"χ² = **{chi2:.2f}** · "
                f"Bậc tự do: **{dof}** · "
                f"p {p_value_text(p_chi)}"
            )

            if p_chi < 0.05:
                st.write(
                    "Có bằng chứng về mối liên hệ "
                    "ở mức ý nghĩa 5%."
                )
            else:
                st.write(
                    "Chưa đủ bằng chứng về mối liên hệ "
                    "ở mức ý nghĩa 5%."
                )

    with c2:
        with st.container(border=True):
            st.subheader("ADR giữa hai nhóm đặt phòng")
            st.caption("KIỂM ĐỊNH WELCH T-TEST")

            st.write(
                f"t = **{t_stat:.2f}** · "
                f"p {p_value_text(p_t)}"
            )

            st.write(
                f"Đơn hủy: **{canceled.mean():.2f} EUR** · "
                f"Không hủy: **{stayed.mean():.2f} EUR**"
            )

            if p_t < 0.05:
                st.write(
                    "ADR trung bình khác nhau "
                    "ở mức ý nghĩa 5%."
                )
            else:
                st.write(
                    "Chưa đủ bằng chứng về khác biệt ADR "
                    "ở mức ý nghĩa 5%."
                )

    st.caption(
        "Kiểm định thể hiện mối liên hệ hoặc khác biệt, "
        "không chứng minh quan hệ nhân quả."
    )


# ==================================================
# 8. TRANG DỰ BÁO HỦY
# ==================================================

elif page == "3. Dự báo hủy":

    try:
        bundle, version_warning = load_model()

    except Exception as error:
        st.error(
            f"Không đọc được model_bundle.pkl: {error}"
        )

        st.info(
            "Đặt mô hình cùng thư mục app.py "
            "và dùng cùng phiên bản scikit-learn với bạn 1."
        )

        st.stop()

    if version_warning:
        st.warning(
            "Phiên bản scikit-learn khác phiên bản lưu mô hình. "
            "Đồng bộ phiên bản trước khi chốt demo."
        )

    with st.form("booking_form"):
        st.subheader("Thông tin lưu trú")

        c1, c2, c3 = st.columns(3)

        lead_time = c1.number_input(
            "Đặt trước (ngày)",
            0, 737, 150
        )

        total_nights = c2.number_input(
            "Số đêm",
            1, 60, 3
        )

        adr = c3.number_input(
            "Giá mỗi đêm (EUR)",
            0.0, 1000.0, 95.0
        )

        adults = c1.number_input(
            "Người lớn",
            1, 10, 2
        )

        children = c2.number_input(
            "Trẻ em",
            0, 10, 0
        )

        babies = c3.number_input(
            "Em bé",
            0, 10, 0
        )

        st.divider()

        st.subheader("Đặc điểm đặt phòng")

        labels = {
            "meal": "Gói ăn",
            "market_segment": "Phân khúc",
            "distribution_channel": "Kênh bán",
            "deposit_type": "Loại đặt cọc",
            "customer_type": "Loại khách",
            "reserved_room_type": "Hạng phòng đặt"
        }

        defaults = {
            "meal": "BB",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "reserved_room_type": "A"
        }

        fields = st.columns(3)
        values = {}

        for i, name in enumerate(bundle["cat_features"]):
            options = sorted(
                df[name].dropna().unique().tolist()
            )

            default = defaults.get(name, options[0])

            if default in options:
                default_index = options.index(default)
            else:
                default_index = 0

            values[name] = fields[i % 3].selectbox(
                labels.get(name, name),
                options,
                index=default_index
            )

        st.divider()

        st.subheader("Lịch sử và thông tin bổ sung")

        c1, c2, c3 = st.columns(3)

        previous_cancellations = c1.number_input(
            "Số lần hủy trước đây",
            0, 30, 0
        )

        previous_ok = c2.number_input(
            "Số đặt phòng không hủy trước đây",
            0, 100, 0
        )

        waiting = c3.number_input(
            "Ngày trong danh sách chờ",
            0, 400, 0
        )

        repeated = c1.checkbox("Khách quay lại")

        has_agent = c2.checkbox(
            "Có đại lý",
            value=True
        )

        has_company = c3.checkbox(
            "Có công ty liên kết"
        )

        st.write("")

        submitted = st.form_submit_button(
            "Dự báo xác suất hủy"
        )

    if submitted:
        values.update(
            lead_time=lead_time,
            total_nights=total_nights,
            adr=adr,
            adults=adults,
            children=children,
            babies=babies,
            total_guests=adults + children + babies,
            adr_zero=int(adr == 0),
            is_repeated_guest=int(repeated),
            has_agent=int(has_agent),
            has_company=int(has_company),
            previous_cancellations=previous_cancellations,
            previous_bookings_not_canceled=previous_ok,
            days_in_waiting_list=waiting
        )

        encoded = pd.get_dummies(
            pd.DataFrame([values]),
            columns=bundle["cat_features"]
        )

        encoded = encoded.reindex(
            columns=bundle["feature_names"],
            fill_value=0
        )

        model = bundle["model"]

        cancel_column = list(
            model.classes_
        ).index(1)

        probability = model.predict_proba(encoded)[
            0, cancel_column
        ]

        st.subheader("Kết quả dự báo")

        c1, c2 = st.columns([1, 2])

        c1.metric(
            "Xác suất hủy dự báo",
            f"{probability:.2%}"
        )

        with c2:
            st.html("""
            <div class="result-note">
                Kết quả áp dụng cho thông tin vừa gửi trong form.
                Khi đổi thông tin, bấm dự báo để tính lại.
            </div>
            """)

    st.caption(
        "Xác suất từ Random Forest chưa hiệu chỉnh calibration. "
        "Thay đổi đầu vào không đo tác động nhân quả "
        "của chính sách."
    )


# ==================================================
# 9. TRANG WHAT-IF
# ==================================================

elif page == "4. What-if":

    st.caption(
        "Chọn một kịch bản bên dưới. "
        "Tham số công suất và hoa hồng nằm ở thanh bên trái."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Hoa hồng kênh bán",
            "Bán vượt phòng · Newsvendor",
            "Bảo vệ phòng · Littlewood"
        ]
    )

    # ----------------------------------------------
    # TAB 1: HOA HỒNG
    # ----------------------------------------------

    with tab1:
        st.subheader("Doanh thu sau hoa hồng")

        stayed = df[df["is_canceled"] == 0].copy()

        stayed["gross"] = (
            stayed["adr"] * stayed["total_nights"]
        )

        stayed["net"] = stayed["gross"] * np.where(
            stayed["distribution_channel"] == "TA/TO",
            1 - commission,
            1
        )

        totals = (
            stayed.groupby("distribution_channel")[
                ["gross", "net"]
            ]
            .sum()
            .reset_index()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Doanh thu gộp (EUR)",
            f"{totals['gross'].sum():,.0f}"
        )

        c2.metric(
            "Sau hoa hồng (EUR)",
            f"{totals['net'].sum():,.0f}"
        )

        c3.metric(
            "Hoa hồng giả định (EUR)",
            f"{totals['gross'].sum() - totals['net'].sum():,.0f}"
        )

        long_data = totals.rename(
            columns={
                "gross": "Doanh thu gộp",
                "net": "Sau hoa hồng"
            }
        ).melt(
            id_vars="distribution_channel",
            var_name="series",
            value_name="value"
        )

        with st.container(border=True):
            st.caption(
                f"Hoa hồng TA/TO: {commission:.0%} · "
                "Các kênh còn lại: 0%"
            )

            chart = (
                alt.Chart(long_data)
                .mark_bar(
                    cornerRadiusTopLeft=4,
                    cornerRadiusTopRight=4
                )
                .encode(
                    x=alt.X(
                        "distribution_channel:N",
                        title=None,
                        axis=alt.Axis(labelAngle=0)
                    ),
                    xOffset=alt.XOffset(
                        "series:N",
                        sort=[
                            "Doanh thu gộp",
                            "Sau hoa hồng"
                        ]
                    ),
                    y=alt.Y(
                        "value:Q",
                        title="Doanh thu (EUR)",
                        axis=alt.Axis(format="~s")
                    ),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(
                            domain=[
                                "Doanh thu gộp",
                                "Sau hoa hồng"
                            ],
                            range=[SKY, BLUE]
                        )
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "distribution_channel:N",
                            title="Kênh"
                        ),
                        alt.Tooltip(
                            "series:N",
                            title="Chỉ số"
                        ),
                        alt.Tooltip(
                            "value:Q",
                            title="EUR",
                            format=",.2f"
                        )
                    ]
                )
                .properties(height=300)
            )

            st.altair_chart(
                chart_style(chart),
                width="stretch"
            )

        st.caption(
            "Áp dụng hoa hồng cho toàn bộ TA/TO làm đại diện OTA. "
            "Doanh thu sau hoa hồng chưa phải lợi nhuận "
            "vì chưa trừ chi phí vận hành."
        )

    # ----------------------------------------------
    # TAB 2: NEWSVENDOR
    # ----------------------------------------------

    with tab2:
        st.subheader("Hạn mức bán vượt phòng")

        st.caption(
            "Xác suất hủy từ Random Forest được tổng hợp theo ngày lưu trú. "
            "Chi phí là tham số kịch bản."
        )

        try:
            daily_risk = load_daily_model_risk(df)
        except (FileNotFoundError, KeyError, ValueError) as error:
            st.error(f"Không tạo được rủi ro hủy từ model: {error}")
        else:
            date_options = daily_risk["arrival_date"].dt.date.tolist()
            default_date_index = int(
                daily_risk["expected_cancellations"].idxmax()
            )
            default_date_index = date_options.index(
                daily_risk.loc[default_date_index, "arrival_date"].date()
            )
            selected_date = st.selectbox(
                "Ngày mô phỏng",
                date_options,
                index=default_date_index,
                format_func=lambda value: value.strftime("%d/%m/%Y")
            )
            selected = daily_risk[
                daily_risk["arrival_date"].dt.date == selected_date
            ].iloc[0]

            mean_cancel = float(selected["expected_cancellations"])
            std_cancel = float(selected["cancellation_std"])
            current_bookings = int(selected["bookings"])

            with st.container(border=True):
                c1, c2 = st.columns(2)

                room_value = c1.number_input(
                    "Chi phí cơ hội phòng trống (EUR)",
                    1.0, 1000.0, float(REVENUE_PARAMS["adr"])
                )

                walk_multiplier = c2.slider(
                    "Chi phí chuyển khách / chi phí phòng trống",
                    1.0, 2.5, 1.5, 0.1
                )

            result = calculate_overbooking(
                mean_cancels=mean_cancel,
                std_cancels=std_cancel,
                adr=room_value,
                walk_cost_multiplier=walk_multiplier,
            )
            extra = result["overbook_rooms"]
            booking_limit = capacity + extra
            remaining_limit = max(0, booking_limit - current_bookings)

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Booking-room-night hiện có",
                f"{current_bookings:,}"
            )

            c2.metric(
                "Hủy kỳ vọng từ model",
                f"{mean_cancel:.1f} ± {std_cancel:.1f}"
            )

            c3.metric(
                "Có thể nhận thêm theo hạn mức",
                f"{remaining_limit:,}"
            )

            with st.expander("Công thức và giả định"):
                st.latex(
                    r"Q = \mu + \Phi^{-1}"
                    r"\left(\frac{C_u}{C_u+C_o}\right)\sigma"
                )

                st.write(
                    f"Cu = {room_value:.2f} EUR; "
                    f"Co = {room_value * walk_multiplier:.2f} EUR."
                )

                st.caption(
                    f"Model dùng các booking từ tập test theo thời gian "
                    f"(từ {pd.Timestamp('2017-01-01').date()}) và giả định mỗi "
                    "booking dùng một phòng mỗi đêm. Hạn mức tổng = công suất "
                    "+ buffer overbooking."
                )

            baseline = calculate_overbooking(
                mean_cancels=REVENUE_PARAMS["mean_cancel"],
                std_cancels=REVENUE_PARAMS["std_cancel"],
                adr=REVENUE_PARAMS["adr"],
                walk_cost_multiplier=REVENUE_PARAMS["walk_cost_multiplier"],
            )
            with st.expander("Kết quả baseline chính thức"):
                st.write(
                    f"EDA bàn giao: μ={REVENUE_PARAMS['mean_cancel']}, "
                    f"σ={REVENUE_PARAMS['std_cancel']}, "
                    f"ADR={REVENUE_PARAMS['adr']:.2f} EUR."
                )
                st.write(
                    f"Kết quả chính thức theo công thức hiện tại: cho phép bán vượt "
                    f"{baseline['overbook_rooms']} phòng. Giá trị Q*="
                    f"{REVENUE_PARAMS['handoff_overbook_rooms']} trong biên bản cũ "
                    "không được sử dụng vì không tái lập được từ bộ tham số này."
                )
                st.caption(
                    "Báo cáo nên sử dụng Q*=33 và ghi chú Q*=28 là giá trị bàn giao "
                    "cũ để tránh trình bày hai đáp án chính thức."
                )

    # ----------------------------------------------
    # TAB 3: LITTLEWOOD
    # ----------------------------------------------

    with tab3:
        st.subheader("Bảo vệ phòng cho nhóm giá cao")

        st.caption(
            "Minh họa một ngày: nhóm giá thấp đặt trước, "
            "nhóm giá cao đến sau."
        )

        with st.container(border=True):
            c1, c2 = st.columns(2)

            high = c1.number_input(
                "Giá nhóm cao / Direct (EUR)",
                1.0, 1000.0, float(REVENUE_PARAMS["p_high"])
            )

            low_gross = c1.number_input(
                "Giá nhóm thấp / TA/TO trước hoa hồng (EUR)",
                1.0, 1000.0, float(REVENUE_PARAMS["p_low_gross"])
            )

            mean_high = c2.number_input(
                "Nhu cầu nhóm giá cao kỳ vọng (phòng)",
                0.0, 400.0, float(REVENUE_PARAMS["mean_demand_high"])
            )

            std_high = c2.number_input(
                "Độ lệch chuẩn nhu cầu nhóm giá cao",
                0.0, 100.0, float(REVENUE_PARAMS["std_demand_high"])
            )

        low_net = low_gross * (1 - commission)

        if low_net >= high:
            st.warning(
                "Giá thấp sau hoa hồng cần nhỏ hơn giá cao "
                "để áp dụng cách chia nhóm này."
            )

        else:
            littlewood = calculate_littlewood(
                p_high=high,
                p_low=low_net,
                mean_demand_high=mean_high,
                std_demand_high=std_high,
                capacity=capacity,
            )
            protection = littlewood["protection_level"]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Giá thấp sau hoa hồng",
                f"{low_net:.2f} EUR"
            )

            c2.metric(
                "Phòng bảo vệ nhóm giá cao",
                protection
            )

            c3.metric(
                "Hạn mức nhóm giá thấp",
                capacity - protection
            )

            with st.expander("Công thức và giả định"):
                st.latex(
                    r"y = \mu_H + \Phi^{-1}"
                    r"\left(1-\frac{p_L}{p_H}\right)\sigma_H"
                )

                st.caption(
                    "Nhu cầu là giả định cho demo. "
                    "pL: giá thấp sau hoa hồng; pH: giá cao; "
                    "μH và σH: kỳ vọng và độ lệch chuẩn "
                    "nhu cầu nhóm giá cao. Littlewood chưa nhận trực tiếp "
                    "xác suất hủy từ Random Forest."
                )

    st.caption(
        "Newsvendor đã dùng rủi ro hủy tổng hợp từ Random Forest. "
        "Littlewood vẫn là mô phỏng riêng vì cần một mô hình dự báo nhu cầu "
        "nhóm giá cao."
    )


# ==================================================
# 10. CHÂN TRANG
# ==================================================

st.html("""
<div class="footer">
    City Hotel · Phân tích dữ liệu lịch sử
    &amp; mô phỏng quản trị doanh thu · Đơn vị tiền tệ: EUR
</div>
""")
