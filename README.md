👨‍⚕️ Trợ lý Sức khỏe Tiểu đường (Diabetes Health Assistant)

Đây là một ứng dụng web xây dựng bằng Streamlit, sử dụng Machine Learning và Generative AI (Google Gemini) để cung cấp hai chức năng chính liên quan đến việc sàng lọc sớm bệnh tiểu đường.

🌟 Tính năng

Ứng dụng được chia làm 2 tab chính:

1. Dự đoán (Model số)

Mục đích: Dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các chỉ số lâm sàng (xét nghiệm máu, đo lường cơ thể).

Công nghệ: Sử dụng các mô hình Machine Learning đã được huấn luyện (LightGBM, Random Forest, Logistic Regression).

Đầu vào: Tuổi, giới tính, huyết áp, bệnh tim, lịch sử hút thuốc, BMI, mức HbA1c, và mức đường huyết.

Đầu ra:

Phân loại nguy cơ (Cao/Thấp).

Xác suất bị bệnh (%).

Top 3 yếu tố nguy cơ hàng đầu ảnh hưởng đến dự đoán (ví dụ: HbA1c, Đường huyết, Tuổi).

2. Phân tích Triệu chứng (Gemini)

Mục đích: Cung cấp một phân tích sơ bộ dựa trên các triệu chứng cơ năng (người dùng tự cảm nhận).

Công nghệ: Sử dụng Google Gemini API (gemini-2.5-flash-preview-09-2025).

Đầu vào: Người dùng tick vào các triệu chứng phổ biến (ví dụ: khát nước nhiều, đi tiểu thường xuyên, mệt mỏi, mờ mắt...) và mô tả thêm.

Đầu ra: Một bản phân tích do AI tạo ra, đánh giá mức độ rủi ro dựa trên triệu chứng và đưa ra khuyến nghị (luôn khuyến nghị đi gặp bác sĩ).

🛠️ Cài đặt & Thiết lập

Để chạy dự án này local, bạn cần làm theo các bước sau:

1. Clone Repository

git clone [URL_REPO_CUA_BAN]
cd [TEN_THU_MUC_CUA_BAN]


2. Tạo môi trường ảo (Khuyến nghị)

# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate


3. Cài đặt thư viện

Cài đặt các thư viện cần thiết có trong app.py:

pip install streamlit pandas requests joblib numpy scikit-learn lightgbm


4. Đặt các file Model

Đảm bảo bạn có thư mục models/ chứa các file .pkl đã được huấn luyện:

models/
├── lightgbm.pkl
├── logistic_regression.pkl
├── random_forest.pkl
└── scaler_lr.pkl


5. Thiết lập API Key (Quan trọng)

Ứng dụng này yêu cầu API Key của Google Gemini để chạy Tab 2.

Tạo thư mục .streamlit trong thư mục gốc của dự án (nếu chưa có).

Tạo file tên là secrets.toml bên trong thư mục .streamlit.

Thêm API key của bạn vào file secrets.toml với nội dung sau:

gemini_api_key = "YOUR_GEMINI_API_KEY_GOES_HERE"


🚀 Chạy ứng dụng

Sau khi hoàn tất cài đặt, chạy lệnh sau trong terminal:

streamlit run app.py


Streamlit sẽ mở một tab trên trình duyệt của bạn (thường là http://localhost:8501).

⚠️ Tuyên bố miễn trừ trách nhiệm y tế

LƯU Ý QUAN TRỌNG:

Ứng dụng này được tạo ra với mục đích tham khảo và giáo dục.

Các dự đoán và phân tích từ cả model Machine Learning và Gemini AI KHÔNG phải là chẩn đoán y tế.

Kết quả từ ứng dụng này TUYỆT ĐỐI KHÔNG thay thế cho việc tư vấn, chẩn đoán, hoặc điều trị từ các chuyên gia y tế có chuyên môn.

Luôn luôn tìm kiếm lời khuyên của bác sĩ hoặc nhà cung cấp dịch vụ y tế đủ điều kiện nếu bạn có bất kỳ câu hỏi nào liên quan đến tình trạng sức khỏe của mình.
