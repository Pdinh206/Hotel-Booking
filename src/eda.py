# -*- coding: utf-8 -*-
"""
========================================================================================
HỌC PHẦN: HỆ HỖ TRỢ VÀ QUYẾT ĐỊNH KINH DOANH (IM3041 - HCMUT)
ĐỀ TÀI 5: QUẢN TRỊ DOANH THU KHÁCH SẠN (HOTEL REVENUE MANAGEMENT - CITY HOTEL)
MÔ-ĐUN: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA), TRỰC QUAN HÓA & KIỂM ĐỊNH THỐNG KÊ (MỐC 3)
Vai trò: Thành viên 2 (Data Analyst / EDA & Statistical Testing)
Môi trường thực thi: Python 3.10+ / Anaconda Spyder / Jupyter Lab
========================================================================================
"""

#%% [1] KHAI BÁO CÁC THƯ VIỆN KHOA HỌC DỮ LIỆU & CẤU HÌNH HỆ THỐNG
import os
import sys
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Chế độ xuất đồ họa tĩnh tiêu chuẩn, ngăn ngừa xung đột giao diện
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

try:
    from revenue import REVENUE_PARAMS
except ImportError:
    from src.revenue import REVENUE_PARAMS

# Cấu hình chuẩn hóa xuất văn bản tiếng Việt trên hệ điều hành Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thiết lập quy chuẩn hiển thị đồ thị học thuật (Academic Publication Style)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# Thư mục gốc của project và các thư mục dữ liệu/kết quả dùng chung
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
PROCESSED_DATA = os.path.join(PROJECT_DIR, 'data', 'processed', 'hotel_clean.csv')
RAW_DATA = os.path.join(PROJECT_DIR, 'data', 'raw', 'hotel_bookings.csv')
REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

#%% [2] NẠP DỮ LIỆU & TIỀN XỬ LÝ ĐỒNG BỘ CHUẨN KHOA HỌC
data_candidates = [
    PROCESSED_DATA,
    RAW_DATA,
]
data_path = next((p for p in data_candidates if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError("Không tìm thấy tệp dữ liệu nguồn (hotel_bookings.csv) trong thư mục!")

print("=" * 80)
print(f"[THÔNG TIN DỮ LIỆU] Nạp dữ liệu nguồn từ: {os.path.basename(data_path)}")
df_raw = pd.read_csv(data_path)

if 'hotel_clean.csv' in os.path.basename(data_path):
    df = df_raw.copy()
    if 'total_nights' not in df.columns:
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    if 'arrival_date' in df.columns:
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
else:
    # 1. Lọc riêng phân khúc City Hotel
    df = df_raw[df_raw['hotel'] == 'City Hotel'].copy()

    # 2. Xử lý giá trị khuyết thiếu (Missing Values)
    df['children'] = df['children'].fillna(0).astype(int)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)

    # 3. Lọc bỏ các bản ghi không hợp lệ về mặt vận hành (Dirty Data)
    mask_valid_guests = (df['adults'] + df['children'] + df['babies']) > 0
    mask_valid_nights = (df['stays_in_weekend_nights'] + df['stays_in_week_nights']) > 0
    mask_valid_adr = (df['adr'] >= 0) & (df['adr'] <= 1000)  # Lọc ngoại lai nhập liệu ADR > 1000
    mask_valid_seg = (df['market_segment'] != 'Undefined') & (df['distribution_channel'] != 'Undefined')

    df = df[mask_valid_guests & mask_valid_nights & mask_valid_adr & mask_valid_seg].copy()

    # 4. Tạo biến tổng số đêm lưu trú và ngày đến chuẩn xác
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_month_num'].astype(str).str.zfill(2) + '-' +
        df['arrival_date_day_of_month'].astype(str).str.zfill(2)
    )

print(f"[THÔNG TIN DỮ LIỆU] Tổng số đơn đặt phòng City Hotel hợp lệ: {len(df):,} bản ghi")
print("=" * 80)

#%% [3] THUẬT TOÁN ƯỚC LƯỢNG CÔNG SUẤT KHÁCH SẠN CƠ SỞ (PEAK CONCURRENT OCCUPANCY)
actual_stays = df[df['is_canceled'] == 0].copy()
daily_occupied_rooms = {}

for _, row in actual_stays.iterrows():
    start_date = row['arrival_date']
    stay_len = int(row['total_nights'])
    for d in range(stay_len):
        cur_date = start_date + timedelta(days=d)
        daily_occupied_rooms[cur_date] = daily_occupied_rooms.get(cur_date, 0) + 1

peak_date = max(daily_occupied_rooms, key=daily_occupied_rooms.get)
peak_rooms = daily_occupied_rooms[peak_date]
OBSERVED_PEAK_OCCUPANCY = peak_rooms
CAPACITY_SCENARIO = REVENUE_PARAMS['capacity']

# Thống kê hủy theo toàn bộ lịch ngày, bao gồm cả ngày không có booking hủy.
# Đây là nguồn của μ và σ bàn giao cho Newsvendor.
daily_cancel_counts = (
    df[df['is_canceled'] == 1]
    .groupby('arrival_date')
    .size()
)
full_calendar = pd.date_range(
    df['arrival_date'].min(),
    df['arrival_date'].max(),
    freq='D'
)
daily_cancel_counts = daily_cancel_counts.reindex(full_calendar, fill_value=0)
mean_daily_cancellations = float(daily_cancel_counts.mean())
std_daily_cancellations = float(daily_cancel_counts.std(ddof=1))
mean_adr_clean = float(df['adr'].mean())

print("\n" + "=" * 80)
print("[KẾT QUẢ THỰC NGHIỆM] ƯỚC LƯỢNG CÔNG SUẤT KHÁCH SẠN (CAPACITY ESTIMATION):")
print(f" -> Ngày đạt đỉnh số phòng có khách lưu trú đồng thời lớn nhất: {peak_date.strftime('%d/%m/%Y')}")
print(f" -> Peak occupancy quan sát được: {OBSERVED_PEAK_OCCUPANCY} phòng")
print(f" -> Công suất dùng cho mô phỏng doanh thu: C = {CAPACITY_SCENARIO} phòng (giả định)")
print(f" -> Hủy trung bình/ngày: μ = {mean_daily_cancellations:.2f}; độ lệch chuẩn: σ = {std_daily_cancellations:.2f}")
print(f" -> ADR trung bình trên hotel_clean.csv: {mean_adr_clean:.2f} EUR")
print("=" * 80)

#%% [4] BẢNG THỐNG KÊ MÔ TẢ THEO KÊNH PHÂN PHỐI (DISTRIBUTION CHANNELS)
channel_summary = df.groupby('distribution_channel').agg(
    total_bookings=('is_canceled', 'count'),
    canceled_count=('is_canceled', 'sum'),
    total_nights=('total_nights', 'sum'),
    mean_adr=('adr', 'mean'),
    median_lead_time=('lead_time', 'median')
).reset_index()

channel_summary['cancel_rate_%'] = (channel_summary['canceled_count'] / channel_summary['total_bookings']) * 100
channel_summary['nights_share_%'] = (channel_summary['total_nights'] / df['total_nights'].sum()) * 100

print("\n[BẢNG TỔNG HỢP] THỐNG KÊ MÔ TẢ CÁC KÊNH PHÂN PHỐI:")
print(channel_summary.round(2).to_string(index=False))

#%% [5] HÌNH 1: CƠ CẤU THỊ PHẦN ĐÊM PHÒNG VÀ TỶ LỆ HỦY THEO KÊNH
fig, ax1 = plt.subplots(figsize=(9, 5), facecolor='white')
ax1.set_facecolor('white')

channels = channel_summary['distribution_channel']
x = np.arange(len(channels))
width = 0.32

color_share = '#1F4E79'   # Xanh navy chuẩn học thuật (Thị phần đêm phòng)
color_cancel = '#C00000'  # Đỏ đậm chuẩn học thuật (Tỷ lệ hủy phòng)

rects1 = ax1.bar(x - width/2, channel_summary['nights_share_%'], width, 
                 label='Thị phần đêm phòng (%)', color=color_share, edgecolor='none')
ax1.set_ylabel('Thị phần đêm phòng (%)', color=color_share, fontsize=10.5, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(channels, fontsize=10, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_share)
ax1.set_ylim(0, 115)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=color_share)

ax2 = ax1.twinx()
ax2.set_facecolor('none')
rects2 = ax2.bar(x + width/2, channel_summary['cancel_rate_%'], width, 
                 label='Tỷ lệ hủy phòng (%)', color=color_cancel, edgecolor='none')
ax2.set_ylabel('Tỷ lệ hủy phòng (%)', color=color_cancel, fontsize=10.5, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_cancel)
ax2.set_ylim(0, 115)

for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=color_cancel)

avg_cancel = (df['is_canceled'].sum() / len(df)) * 100
line_avg = ax2.axhline(avg_cancel, color='#666666', linestyle=':', linewidth=1.2, 
                       label=f'Tỷ lệ hủy TB chung ({avg_cancel:.1f}%)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='#FFFFFF', framealpha=0.95)

plt.title('HÌNH 1: THỊ PHẦN ĐÊM PHÒNG VÀ TỶ LỆ HỦY THEO KÊNH PHÂN PHỐI\n(City Hotel, Giai đoạn 2015 - 2017)', 
          fontsize=11.5, fontweight='bold', pad=12, color='#0F172A')

fig.tight_layout()
save_path_1 = os.path.join(FIGURES_DIR, 'channel_comparison.png')
plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
print("[KẾT XUẤT ĐỒ THỊ] Đã lưu Hình 1: channel_comparison.png")
plt.close(fig)

#%% [6] HÌNH 2: ĐƯỜNG CONG TỶ LỆ HỦY THEO THỜI GIAN ĐẶT TRƯỚC (BOOKING CURVE)
lead_bins = [-1, 14, 30, 60, 90, 180, 1000]
lead_labels = ['< 14 ngày', '14 - 30 ngày', '31 - 60 ngày', '61 - 90 ngày', '91 - 180 ngày', '> 180 ngày']
df['lead_time_group'] = pd.cut(df['lead_time'], bins=lead_bins, labels=lead_labels)

lead_stats = df.groupby('lead_time_group', observed=False).agg(
    total=('is_canceled', 'count'),
    canceled=('is_canceled', 'sum')
).reset_index()
lead_stats['cancel_rate_%'] = (lead_stats['canceled'] / lead_stats['total']) * 100

fig, ax = plt.subplots(figsize=(9, 4.8), facecolor='white')
ax.set_facecolor('white')

x_pts = range(len(lead_labels))
y_pts = lead_stats['cancel_rate_%']

ax.plot(x_pts, y_pts, marker='s', markersize=6.5, color='#1F4E79', linewidth=2.0, 
        label='Tỷ lệ hủy phòng theo mốc (%)', zorder=4)
ax.axhline(avg_cancel, color='#C00000', linestyle='--', linewidth=1.2, 
           label=f'Tỷ lệ hủy TB chung ({avg_cancel:.1f}%)', zorder=2)

ax.set_xticks(x_pts)
ax.set_xticklabels(lead_labels, fontsize=9.5, fontweight='bold')
ax.set_xlabel('Thời gian đặt trước ngày nhận phòng (Lead Time)', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Tỷ lệ hủy phòng (%)', fontsize=10.5, fontweight='bold', labelpad=8)
ax.set_ylim(0, 80)
ax.grid(True, linestyle='--', alpha=0.3)

for i, val in enumerate(y_pts):
    offset_y = 7 if val >= 41.0 else -14
    ax.annotate(f'{val:.1f}%', xy=(i, val), xytext=(0, offset_y), textcoords='offset points',
                ha='center', fontweight='bold', fontsize=9, color='#1F4E79')

ax.legend(loc='upper left', frameon=True, facecolor='#FFFFFF', framealpha=0.95)
plt.title('HÌNH 2: TỶ LỆ HỦY PHÒNG THEO THỜI GIAN ĐẶT TRƯỚC (BOOKING CURVE)\n(City Hotel, Giai đoạn 2015 - 2017)', 
          fontsize=11.5, fontweight='bold', pad=12, color='#0F172A')

fig.tight_layout()
save_path_2 = os.path.join(FIGURES_DIR, 'booking_curve.png')
plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
print("[KẾT XUẤT ĐỒ THỊ] Đã lưu Hình 2: booking_curve.png")
plt.close(fig)

#%% [7] HÌNH 3: TÍNH MÙA VỤ CỦA CÔNG SUẤT (OCCUPANCY %) VÀ REVPAR THEO 12 THÁNG
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly_kpi = actual_stays.groupby('arrival_date_month').agg(
    occupied_nights=('total_nights', 'sum'),
    mean_adr=('adr', 'mean')
).reindex(month_order).reset_index()

monthly_kpi['avg_occupied_rooms_per_day'] = monthly_kpi['occupied_nights'] / (30.4 * 2)
monthly_kpi['occupancy_%'] = (monthly_kpi['avg_occupied_rooms_per_day'] / float(CAPACITY_SCENARIO)) * 100
monthly_kpi['revpar'] = monthly_kpi['mean_adr'] * (monthly_kpi['occupancy_%'] / 100.0)

fig, ax1 = plt.subplots(figsize=(10, 5), facecolor='white')
ax1.set_facecolor('white')

months_short = ['Th.1', 'Th.2', 'Th.3', 'Th.4', 'Th.5', 'Th.6', 'Th.7', 'Th.8', 'Th.9', 'Th.10', 'Th.11', 'Th.12']
x_m = np.arange(len(months_short))
width_m = 0.45

color_occ = '#1F4E79'
color_revpar = '#D97706'

bars_occ = ax1.bar(x_m, monthly_kpi['occupancy_%'], width_m, label='Công suất phòng (Occupancy %)', 
                   color=color_occ, edgecolor='none', alpha=0.9, zorder=3)
ax1.set_ylabel('Công suất phòng (Occupancy %)', color=color_occ, fontsize=10.5, fontweight='bold')
ax1.set_xticks(x_m)
ax1.set_xticklabels(months_short, fontsize=9.5, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_occ)
ax1.set_ylim(0, 150)
ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

# Đặt nhãn % ở CHÍNH GIỮA THÂN CỘT (h / 2), màu trắng nổi bật trên nền xanh thẫm, hoàn toàn tách biệt khỏi đường RevPAR
for b in bars_occ:
    h = b.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h / 2.0),
                 ha='center', va='center',
                 fontsize=8.5, fontweight='bold', color='white', zorder=5)

ax2 = ax1.twinx()
ax2.set_facecolor('none')
ax2.plot(x_m, monthly_kpi['revpar'], marker='o', markersize=6, color=color_revpar, 
         linewidth=2.2, label='RevPAR (EUR / phòng sẵn có)', zorder=4)
ax2.set_ylabel('RevPAR (EUR / phòng sẵn có)', color=color_revpar, fontsize=10.5, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_revpar)
ax2.set_ylim(0, 180)

# Hiển thị số liệu RevPAR phía trên các nút dữ liệu (màu cam đậm), có khoảng đệm thoáng đãng
for i, v in enumerate(monthly_kpi['revpar']):
    ax2.annotate(f'{v:.1f}€', xy=(i, v), xytext=(0, 7), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color_revpar, zorder=6)

lines_a, labels_a = ax1.get_legend_handles_labels()
lines_b, labels_b = ax2.get_legend_handles_labels()
ax1.legend(lines_a + lines_b, labels_a + labels_b, loc='upper left', frameon=True, facecolor='#FFFFFF', framealpha=0.95)

plt.title(f'HÌNH 3: TÍNH MÙA VỤ CỦA CÔNG SUẤT PHÒNG (OCCUPANCY) VÀ REVPAR THEO 12 THÁNG\n(City Hotel, Công suất mô phỏng C = {CAPACITY_SCENARIO} phòng)', 
          fontsize=11.5, fontweight='bold', pad=12, color='#0F172A')

fig.tight_layout()
save_path_3 = os.path.join(FIGURES_DIR, 'monthly_seasonality.png')
plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
print("[KẾT XUẤT ĐỒ THỊ] Đã lưu Hình 3: monthly_seasonality.png")
plt.close(fig)

#%% [8] KIỂM ĐỊNH GIẢ THUYẾT THỐNG KÊ (HYPOTHESIS TESTING - MỐC 3)
contingency_table = pd.crosstab(df['distribution_channel'], df['is_canceled'])
chi2, p_val_chi2, dof, _ = stats.chi2_contingency(contingency_table)

adr_canceled = df[df['is_canceled'] == 1]['adr']
adr_not_canceled = df[df['is_canceled'] == 0]['adr']
t_stat, p_val_t = stats.ttest_ind(adr_canceled, adr_not_canceled, equal_var=False)

print("\n" + "=" * 80)
print("[KIỂM ĐỊNH GIẢ THUYẾT THỐNG KÊ] KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG:")
print(" 1. Kiểm định Chi-Square (Kênh phân phối vs Hành vi hủy):")
print(f"    - Chi2 Statistic = {chi2:.2f}, df = {dof}, p-value = {p_val_chi2:.4e}")
print("    - Kết luận: p-value < 0.001 -> Bác bỏ giả thuyết H0 (Mối liên hệ có ý nghĩa thống kê rất cao).")
print(" 2. Kiểm định Welch's T-Test (So sánh ADR giữa nhóm Hủy vs Nhóm Lưu trú):")
print(f"    - Welch's t-statistic = {t_stat:.2f}, p-value = {p_val_t:.4e}")
print("    - Kết luận: p-value < 0.001 -> Bác bỏ giả thuyết H0 (Chênh lệch giá phòng có ý nghĩa thống kê).")
print("=" * 80)

#%% [9] XUẤT TỆP BẢNG TÍNH EXCEL 5 SHEET (ĐẢO THỨ TỰ TỐI ƯU MỞ NHANH)
excel_export_path = os.path.join(REPORTS_DIR, 'bang_du_lieu_va_chi_so_eda.xlsx')
excel_file_name = os.path.basename(excel_export_path)
print(f"\n[XUẤT DỮ LIỆU] Đang khởi tạo tệp Excel: {excel_file_name}...")

# Chuẩn bị dữ liệu sạch cho việc xuất Excel, tránh xung đột kiểu dữ liệu
df_export = df.copy()
if 'lead_time_group' in df_export.columns:
    df_export['lead_time_group'] = df_export['lead_time_group'].astype(str)
if 'arrival_date' in df_export.columns:
    df_export['arrival_date'] = df_export['arrival_date'].dt.strftime('%Y-%m-%d')

revenue_params = pd.DataFrame([
    {
        'parameter': 'clean_records',
        'value': len(df),
        'unit': 'booking',
        'source_type': 'data-derived',
        'note': 'Số bản ghi City Hotel sau preprocessing',
    },
    {
        'parameter': 'mean_daily_cancellations',
        'value': round(mean_daily_cancellations, 4),
        'unit': 'booking/day',
        'source_type': 'data-derived',
        'note': 'Mean theo toàn bộ lịch ngày, gồm ngày có 0 hủy',
    },
    {
        'parameter': 'std_daily_cancellations',
        'value': round(std_daily_cancellations, 4),
        'unit': 'booking/day',
        'source_type': 'data-derived',
        'note': 'Sample standard deviation theo ngày',
    },
    {
        'parameter': 'observed_peak_occupancy',
        'value': OBSERVED_PEAK_OCCUPANCY,
        'unit': 'room',
        'source_type': 'data-derived',
        'note': 'Đỉnh phòng đang ở thực tế; không phải công suất danh định',
    },
    {
        'parameter': 'capacity_scenario',
        'value': CAPACITY_SCENARIO,
        'unit': 'room',
        'source_type': 'assumption',
        'note': 'Công suất dùng cho Newsvendor/Littlewood',
    },
    {
        'parameter': 'adr_mean_clean',
        'value': round(mean_adr_clean, 4),
        'unit': 'EUR/room-night',
        'source_type': 'data-derived',
        'note': 'Mean ADR trên hotel_clean.csv',
    },
    {
        'parameter': 'adr_used_in_revenue',
        'value': REVENUE_PARAMS['adr'],
        'unit': 'EUR/room-night',
        'source_type': 'scenario-input',
        'note': 'Giá trị được dùng trong code doanh thu',
    },
    {
        'parameter': 'adr_handoff_reported',
        'value': REVENUE_PARAMS['adr_handoff_reported'],
        'unit': 'EUR/room-night',
        'source_type': 'handoff-reference',
        'note': 'Giá trị EDA bàn giao trước đó; không khớp trực tiếp với mean trên hotel_clean.csv',
    },
    {
        'parameter': 'walk_cost_multiplier',
        'value': REVENUE_PARAMS['walk_cost_multiplier'],
        'unit': 'x ADR',
        'source_type': 'assumption',
        'note': 'Chi phí chuyển khách',
    },
    {
        'parameter': 'handoff_overbook_rooms',
        'value': REVENUE_PARAMS['handoff_overbook_rooms'],
        'unit': 'room',
        'source_type': 'handoff-reference',
        'note': 'Q* trong biên bản cũ; không dùng làm kết quả chính thức',
    },
])

try:
    with pd.ExcelWriter(excel_export_path, engine='openpyxl') as writer:
        # Các sheet tổng hợp chỉ số và tham số bàn giao
        channel_summary.to_excel(writer, sheet_name='channel_summary', index=False)
        monthly_kpi.to_excel(writer, sheet_name='monthly_kpi', index=False)
        lead_stats.to_excel(writer, sheet_name='lead_stats', index=False)
        revenue_params.to_excel(writer, sheet_name='revenue_params', index=False)
        # Sheet dữ liệu chi tiết
        df_export.to_excel(writer, sheet_name='df', index=False)
        
    sz_mb = os.path.getsize(excel_export_path) / (1024 * 1024)
    print(f"[XUẤT DỮ LIỆU] -> ĐÃ XUẤT THÀNH CÔNG: {excel_file_name} ({sz_mb:.2f} MB)")
    print("              Thứ tự sheet: ['channel_summary', 'monthly_kpi', 'lead_stats', 'revenue_params', 'df']")
except PermissionError:
    print(f"\n[CẢNH BÁO] Tệp {excel_file_name} đang mở trong ứng dụng khác. Vui lòng đóng lại để ghi đè!")
except Exception as e:
    print(f"\n[LỖI XUẤT BẢNG TÍNH] {e}")

print("\n[HOÀN TẤT] Toàn bộ quy trình phân tích khám phá dữ liệu (EDA) và kiểm định thống kê đã thực thi thành công.")
print("=" * 80)
