# City Hotel — Dự báo hủy phòng và quản trị doanh thu

Project gồm ba phần dùng chung một nguồn dữ liệu:

- Tiền xử lý dữ liệu và huấn luyện mô hình dự báo hủy phòng.
- Phân tích khám phá dữ liệu (EDA), biểu đồ và kiểm định thống kê.
- Dashboard Streamlit để xem chỉ số, dự báo và mô phỏng What-if.

## Cấu trúc thư mục

```text
├── app/app.py                         # Dashboard Streamlit
├── src/pipeline.py                    # Tiền xử lý + huấn luyện model
├── src/eda.py                         # EDA + xuất biểu đồ/Excel
├── src/revenue.py                     # Hàm dùng chung cho doanh thu
├── notebooks/                         # Notebook của toàn bộ quy trình
├── data/raw/hotel_bookings.csv        # Dữ liệu gốc
├── data/processed/hotel_clean.csv    # Dữ liệu City Hotel đã làm sạch
├── models/model_bundle.pkl            # Random Forest đã huấn luyện
├── reports/                           # Excel và kết quả phân tích
│   ├── feature_importance.csv
│   ├── daily_cancellation_risk.csv    # Rủi ro hủy theo ngày từ Random Forest
│   ├── bang_du_lieu_va_chi_so_eda.xlsx
│   └── figures/
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

## Cài đặt

Yêu cầu Python 3.10+.

Mở PowerShell tại thư mục project:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy pipeline

Pipeline đọc `data/raw/hotel_bookings.csv` và xuất lại dữ liệu, model:

```powershell
python src/pipeline.py
```

Kết quả:

- `data/processed/hotel_clean.csv`
- `models/model_bundle.pkl`
- `reports/feature_importance.csv`
- `reports/daily_cancellation_risk.csv`

File `daily_cancellation_risk.csv` được tạo từ xác suất hủy của Random Forest
trên tập kiểm tra theo thời gian. Các xác suất này được tổng hợp theo ngày lưu
trú để làm đầu vào cho mô phỏng Newsvendor.

Các tham số thực nghiệm do phần EDA bàn giao được tập trung trong
`src/revenue.py` (`REVENUE_PARAMS`). Dashboard hiển thị cả kịch bản
model-driven và baseline EDA để đối chiếu.

## Chạy EDA

EDA đọc dữ liệu đã làm sạch, sau đó xuất Excel và ba biểu đồ vào `reports/`.
Sheet `revenue_params` phân biệt rõ giá trị tính từ dữ liệu và giá trị giả định
dùng cho mô phỏng doanh thu:

```powershell
python src/eda.py
```

Kết quả:

- `reports/bang_du_lieu_va_chi_so_eda.xlsx`
- `reports/figures/channel_comparison.png`
- `reports/figures/booking_curve.png`
- `reports/figures/monthly_seasonality.png`

## Chạy notebook 

Notebook `01_preprocessing_visualized.ipynb` chứa toàn bộ quy trình có thể theo dõi
trực tiếp: đọc dữ liệu, preprocessing, EDA/KPI/kiểm định, Random Forest, Newsvendor,
Littlewood, dự báo đơn mới và xuất các báo cáo. Mở PowerShell tại thư mục `notebooks`:

```powershell
Set-Location notebooks
..\venv\Scripts\python.exe -m notebook 01_preprocessing_visualized.ipynb
```

## Chạy dashboard

```powershell
python -m streamlit run app/app.py
```
Dashboard sử dụng:

- `data/processed/hotel_clean.csv`
- `models/model_bundle.pkl`
- `reports/bang_du_lieu_va_chi_so_eda.xlsx`

Các trang gồm tổng quan, phân tích EDA, dự báo xác suất hủy và What-if cho
hoa hồng, Newsvendor và Littlewood. Tab Newsvendor sử dụng rủi ro hủy được
tổng hợp từ Random Forest; tab Littlewood vẫn là mô phỏng riêng vì cần thêm
một dự báo nhu cầu cho nhóm giá cao.

## Kết quả chính

- Dữ liệu City Hotel sau làm sạch: khoảng 78.894 lượt đặt.
- Tỷ lệ hủy: khoảng 41,91%.
- Random Forest ROC-AUC: khoảng 0,8117.
- Peak occupancy quan sát được: 226 phòng.
- Công suất kịch bản dùng cho mô phỏng: 250 phòng (giả định).

Newsvendor sử dụng giả định chi phí phòng trống, chi phí chuyển khách và giả
định mỗi booking dùng một phòng mỗi đêm. Littlewood sử dụng nhu cầu nhóm giá
cao do người dùng nhập. Xác suất Random Forest chưa được calibration và thay
đổi đầu vào không chứng minh quan hệ nhân quả.

Kết luận: với μ = 41,7; σ = 35,2; ADR thực tế khoảng 105,79 EUR và
hệ số Walk Cost = 1,5, công thức Newsvendor cho Q* = 33 phòng. Q* = 28 chỉ là
giá trị bàn giao, không được dùng vì không tái lập được từ bộ tham số hiện tại.
Giá trị ADR 106,02 EUR cũng chỉ là giá trị bàn giao, không coi là kết quả trực
tiếp từ `hotel_clean.csv`.
