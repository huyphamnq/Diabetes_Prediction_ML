import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io
import requests  # Thư viện để gọi API

# ------------------------
# 1. Load models (cached)
# ------------------------
@st.cache_resource
def load_models():
    """
    Tải các model đã train và scaler.
    Sử dụng cache của Streamlit để chỉ tải 1 lần.
    """
    try:
        scaler = joblib.load('models/scaler_lr.pkl')
        lgb_model = joblib.load('models/lightgbm.pkl')
        lr_model = joblib.load('models/logistic_regression.pkl')
        rf_model = joblib.load('models/random_forest.pkl')
        return scaler, lgb_model, lr_model, rf_model
    except FileNotFoundError:
        st.error("Không tìm thấy file model (.pkl) trong thư mục 'models/'.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        st.stop()

scaler, lgb_model, lr_model, rf_model = load_models()

# ------------------------
# 1.5. Gemini API Function
# ------------------------
def call_gemini(api_key, symptoms_prompt):
    """
    Hàm gọi API của Gemini để phân tích triệu chứng.
    """
    system_prompt = """
    Bạn là một trợ lý y tế AI. Nhiệm vụ của bạn là phân tích các triệu chứng do người dùng cung cấp
    và đưa ra đánh giá sơ bộ về khả năng chúng liên quan đến bệnh tiểu đường.

    QUY TẮC QUAN TRỌNG:
    1. Phân tích các triệu chứng (ví dụ: khát nước nhiều, đi tiểu thường xuyên, mệt mỏi, mờ mắt, sụt cân không rõ nguyên nhân).
    2. Đưa ra đánh giá sơ bộ về mức độ rủi ro (ví dụ: thấp, trung bình, cao) dựa trên các triệu chứng kinh điển.
    3. KHÔNG BAO GIỜ được chẩn đoán.
    4. LUÔN LUÔN kết thúc bằng một khuyến cáo rõ ràng: "Phân tích này chỉ mang tính chất tham khảo, không thay thế chẩn đoán y tế. Bạn CẦN GẶP BÁC SĨ để được xét nghiệm và tư vấn chính xác."

    Định dạng câu trả lời (dùng markdown):
    - **Phân tích triệu chứng:** (Phân tích của bạn về các triệu chứng được cung cấp)
    - **Đánh giá sơ bộ:** (Đánh giá mức độ rủi ro liên quan đến tiểu đường)
    - **Khuyến nghị quan trọng:** (Luôn chốt hạ bằng câu "Phân tích này chỉ mang tính chất tham khảo...")
    """

    model = "gemini-2.5-flash-preview-09-2025"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {"parts": [{"text": symptoms_prompt}]} # Prompt đã được format từ checkboxes
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        # Thêm timeout 30 giây để tránh app bị treo
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        # Tự động báo lỗi nếu API trả về 4xx hoặc 5xx
        response.raise_for_status() 
        
        result = response.json()
        
        # Kiểm tra lỗi trong body (ví dụ: API key sai)
        if 'error' in result:
            return f"LỖI: {result['error']['message']}"
        
        # Kiểm tra nếu API không trả về kết quả (có thể do nội dung bị chặn)
        if 'candidates' not in result or not result['candidates']:
             return "LỖI: API không trả về kết quả. Có thể nội dung của bạn đã bị bộ lọc an toàn chặn."

        text = result['candidates'][0]['content']['parts'][0]['text']
        return text
        
    except requests.exceptions.HTTPError as http_err:
        return f"LỖI HTTP: {http_err} - Vui lòng kiểm tra lại API Key và đảm bảo nó còn hoạt động."
    except requests.exceptions.Timeout:
        return "LỖI: Yêu cầu tới Gemini bị quá thời gian (timeout). Vui lòng thử lại."
    except requests.exceptions.RequestException as e:
        return f"LỖI KẾT NỐI: {e}"
    except Exception as e:
        return f"LỖI KHÔNG XÁC ĐỊNH: {e}"

# ------------------------
# 2. Streamlit Page Setup
# ------------------------
st.set_page_config(layout="wide")
st.title("👨‍⚕️ Trợ lý Sức khỏe Tiểu đường")
st.write("Sử dụng các tab bên dưới để dự đoán bằng model hoặc phân tích triệu chứng bằng AI.")

# ------------------------
# 3. Sidebar Input (Cho Tab 1)
# ------------------------
st.sidebar.header("Thông số (Model dự đoán)")
gender = st.sidebar.selectbox('Giới tính', ('Female', 'Male', 'Other'))
age = st.sidebar.number_input('Tuổi', 0, 120, 30)
hypertension = st.sidebar.radio('Có bị tăng huyết áp không?', ('Không', 'Có'))
heart_disease = st.sidebar.radio('Có bệnh tim không?', ('Không', 'Có'))
smoking_history = st.sidebar.selectbox(
    'Lịch sử hút thuốc', ('never', 'former', 'current', 'not current', 'ever', 'No Info')
)
bmi = st.sidebar.number_input('BMI', 10.0, 70.0, 25.0, format="%.1f")
hba1c_level = st.sidebar.number_input('Mức HbA1c', 3.0, 15.0, 5.7, format="%.1f")
blood_glucose_level = st.sidebar.number_input('Mức đường huyết (mg/dL)', 50, 300, 100)

st.sidebar.header("Chọn Model (Tab 1)")
model_choice = st.sidebar.selectbox(
    "Chọn model",
    ("LightGBM (Khuyến nghị)", "Random Forest", "Logistic Regression(Không khuyến nghị)")
)

# --- TẠO GIAO DIỆN TAB ---
tab1, tab2 = st.tabs(["👨‍⚕️ Dự đoán (Model số)", "🤖 Phân tích Triệu chứng (Gemini)"])

# --- NỘI DUNG TAB 1: MODEL DỰ ĐOÁN ---
with tab1:
    st.header("Dự đoán dựa trên chỉ số lâm sàng")
    st.write("Nhập thông số của bạn ở thanh bên trái và nhấn nút để dự đoán.")

    # ------------------------
    # 4. Preprocess Input
    # ------------------------
    def preprocess_input(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level):
        """
        Chuyển đổi input từ sidebar thành DataFrame 1 hàng
        khớp với các cột mà model đã dùng khi train.
        """
        # Fix: Tên cột phải khớp với model đã train (dùng " not current")
        all_cols = [
            'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
            'blood_glucose_level', 'gender_Male', 'gender_Other',
            'smoking_history_current', 'smoking_history_ever', 'smoking_history_former',
            'smoking_history_never', 'smoking_history_not current'
        ]
        df = pd.DataFrame(columns=all_cols, index=[0]).fillna(0)
        df['age'] = age
        df['hypertension'] = 1 if hypertension == 'Có' else 0
        df['heart_disease'] = 1 if heart_disease == 'Có' else 0
        df['bmi'] = bmi
        df['HbA1c_level'] = hba1c_level
        df['blood_glucose_level'] = blood_glucose_level

        # One-hot encoding thủ công cho các cột categorical
        if gender == 'Male': df['gender_Male'] = 1
        elif gender == 'Other': df['gender_Other'] = 1

        if smoking_history == 'current': df['smoking_history_current'] = 1
        elif smoking_history == 'ever': df['smoking_history_ever'] = 1
        elif smoking_history == 'former': df['smoking_history_former'] = 1
        elif smoking_history == 'never': df['smoking_history_never'] = 1
        # Fix: Tên cột phải khớp với model đã train (dùng " not current")
        elif smoking_history == 'not current': df['smoking_history_not current'] = 1

        return df

    input_df = preprocess_input(gender, age, hypertension, heart_disease,
                                smoking_history, bmi, hba1c_level, blood_glucose_level)

    st.subheader("Dữ liệu đầu vào (Đã xử lý cho model)")
    st.dataframe(input_df)

    # ------------------------
    # 5. Risk Factor Logic
    # ------------------------
    def get_risk_factors(model, input_df, model_choice):
        """
        Chạy dự đoán và lấy ra các yếu tố nguy cơ hàng đầu.
        Xử lý riêng cho LR (cần scale) và Tree-based (không cần).
        """
        if model_choice == "LightGBM (Khuyến nghị)":
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0,1]
            feat_importance = pd.Series(model.feature_importances_, index=input_df.columns)
        elif model_choice == "Random Forest":
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0,1]
            feat_importance = pd.Series(model.feature_importances_, index=input_df.columns)
        else:  # Logistic Regression
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0,1]
            feat_importance = pd.Series(np.abs(model.coef_[0]), index=input_df.columns)

        # Định nghĩa các ngưỡng rủi ro
        # Fix: Tên cột phải khớp với model đã train (dùng " not current")
        thresholds = {
            'age': 50, 'bmi': 25.0, 'HbA1c_level': 5.7, 'blood_glucose_level': 100,
            'hypertension': 1, 'heart_disease': 1,
            'smoking_history_current': 1, 'smoking_history_former': 1,
            'smoking_history_ever': 1, 'smoking_history_not current': 1
        }

        user_values = input_df.iloc[0]
        # Tìm các yếu tố mà người dùng vượt ngưỡng rủi ro
        potential_risks = [f for f,t in thresholds.items() if f in user_values and user_values[f] >= t]

        if potential_risks:
            # Sắp xếp các yếu tố rủi ro đó theo độ quan trọng (feature importance)
            risk_importances = feat_importance[potential_risks].sort_values(ascending=False)
            top_risks = risk_importances.head(3)
        else:
            top_risks = pd.Series(dtype=float)

        return pred, proba, feat_importance, top_risks

    # ------------------------
    # 6. Predict Button
    # ------------------------
    if st.button("Thực hiện Dự đoán"):
        with st.spinner("Đang tính toán..."):
            model_map = {
                "LightGBM (Khuyến nghị)": lgb_model,
                "Random Forest": rf_model,
                "Logistic Regression(Không khuyến nghị)": lr_model
            }
            
            # Xử lý trường hợp key của model_map
            chosen_model_key = model_choice

            pred, proba, feat_importance, top_risks = get_risk_factors(model_map[model_choice], input_df, model_choice)


            # Result display
            col1, col2 = st.columns(2)
            # Dùng để hiển thị tên đẹp (human-readable)
            # Fix: Tên cột phải khớp với model đã train (dùng " not current")
            factor_desc = {
                'age': "Tuổi", 'bmi': "BMI", 'HbA1c_level': "HbA1c",
                'blood_glucose_level': "Đường huyết", 'hypertension': "Tăng huyết áp",
                'heart_disease': "Bệnh tim", 'smoking_history_current': "Đang hút thuốc",
                'smoking_history_former': "Từng hút thuốc", 'smoking_history_ever': "Từng hút thuốc",
                'smoking_history_not current': "Hút thuốc (không phải hiện tại)"
            }

            with col1:
                st.subheader(f"Kết quả Dự đoán ({model_choice})")
                st.metric("Xác suất bị tiểu đường", f"{proba*100:.2f} %")
                if pred == 1: st.error("Nguy cơ: CAO")
                else: st.success("Nguy cơ: THẤP")
                st.bar_chart(pd.DataFrame([[1-proba, proba]], columns=['Không bệnh','Có bệnh']).T)

            with col2:
                if not top_risks.empty:
                    st.write("**Top 3 yếu tố nguy cơ:**")
                    max_imp = top_risks.max()
                    for f, imp in top_risks.items():
                        st.markdown(f"**{factor_desc.get(f,f)}**: {input_df.iloc[0][f]}")
                        st.progress(int(imp/max_imp*100) if max_imp>0 else 0)
                elif pred == 1:
                    st.warning("Model dự đoán có nguy cơ, nhưng không xác định được yếu tố rủi ro chính từ ngưỡng.")
                else:
                    st.info("Không phát hiện yếu tố rủi ro nào trong các ngưỡng đã cài đặt.")

                with st.expander("Xem chi tiết độ quan trọng của tất cả đặc trưng"):
                    st.dataframe(feat_importance.sort_values(ascending=False).to_frame("Importance Score"))

            # Export CSV
            result = input_df.copy()
            result["Model"] = model_choice
            result["Predicted_Risk"] = pred
            result["Probability (%)"] = proba*100
            result["Top_Risk_Factors"] = ", ".join(top_risks.index)
            csv_buf = io.StringIO()
            result.to_csv(csv_buf, index=False)
            st.download_button("Tải kết quả dự đoán ra CSV",
                               csv_buf.getvalue(),
                               file_name=f"prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")

# --- NỘI DUNG TAB 2: PHÂN TÍCH GEMINI ---
with tab2:
    st.header("Phân tích Triệu chứng với Gemini")
    st.write("Sử dụng AI của Google để phân tích các triệu chứng bạn nhập vào và đánh giá sơ bộ nguy cơ tiểu đường.")
    
    # Đọc key từ st.secrets
    try:
        api_key = st.secrets["gemini_api_key"]
        if not api_key: # Trường hợp key có tồn tại nhưng rỗng
            st.warning("Tìm thấy 'gemini_api_key' trong secrets.toml nhưng giá trị bị rỗng.")
            api_key = None
            
    except KeyError:
        # Lỗi nếu không có key 'gemini_api_key' trong file
        st.error("Không tìm thấy 'gemini_api_key' trong file `.streamlit/secrets.toml`.")
        st.info("Vui lòng tạo file `.streamlit/secrets.toml` và thêm vào dòng: `gemini_api_key = \"YOUR_KEY_HERE\"`")
        api_key = None
    except Exception as e:
        # Bắt các lỗi khác nếu st.secrets không tồn tại (hiếm)
        st.error(f"Lỗi khi đọc secrets.toml: {e}")
        api_key = None

    
    st.subheader("Chọn các triệu chứng bạn đang gặp phải:")
    
    # Định nghĩa các triệu chứng
    symptom_list = {
        "polyuria": "Đi tiểu thường xuyên (đặc biệt là ban đêm)",
        "polydipsia": "Khát nước nhiều (khát bất thường)",
        "polyphagia": "Thường xuyên cảm thấy đói (ăn nhiều)",
        "weight_loss": "Sụt cân không rõ nguyên nhân",
        "fatigue": "Mệt mỏi, uể oải, thiếu năng lượng",
        "blurred_vision": "Mờ mắt, thị lực giảm sút",
        "slow_healing": "Vết thương, vết xước lâu lành",
        "infections": "Hay bị nhiễm trùng (da, nướu, ...)",
        "tingling": "Tê bì hoặc ngứa ran ở tay/chân"
    }
    
    # Dùng dictionary để lưu trạng thái của checkboxes
    symptom_states = {}
    
    # Chia cột để giao diện đẹp hơn
    col1, col2, col3 = st.columns(3)
    
    # Chia danh sách triệu chứng ra 3 cột
    symptom_items = list(symptom_list.items())
    items_per_col = (len(symptom_items) + 2) // 3 # Chia (9/3 = 3)
    
    with col1:
        for key, desc in symptom_items[:items_per_col]:
            symptom_states[key] = st.checkbox(desc, key=f"cb_{key}")
            
    with col2:
        for key, desc in symptom_items[items_per_col : 2 * items_per_col]:
            symptom_states[key] = st.checkbox(desc, key=f"cb_{key}")
            
    with col3:
        for key, desc in symptom_items[2 * items_per_col :]:
            symptom_states[key] = st.checkbox(desc, key=f"cb_{key}")

    # Ô nhập các triệu chứng khác
    other_symptoms = st.text_area("Mô tả các triệu chứng khác (nếu có):", 
                                  placeholder="Ví dụ: Da khô, ngứa...", height=100)
    
    if st.button("Phân tích Triệu chứng"):
        # Sửa lại logic kiểm tra key
        if not api_key:
            st.error("Không thể thực hiện. Vui lòng kiểm tra lại API Key trong file secrets.toml.")
        else:
            # Thu thập các triệu chứng đã check
            checked_symptoms = []
            for key, checked in symptom_states.items():
                if checked:
                    checked_symptoms.append(symptom_list[key]) # Lấy mô tả đầy đủ
            
            # Format lại thành một prompt rõ ràng cho Gemini
            final_symptom_prompt = "Dựa trên các thông tin sau, hãy phân tích nguy cơ tiểu đường:\n"
            
            if checked_symptoms:
                final_symptom_prompt += "\nCác triệu chứng đã chọn:\n"
                for s in checked_symptoms:
                    final_symptom_prompt += f"- {s}\n"
            
            if other_symptoms:
                final_symptom_prompt += "\nTriệu chứng khác (do người dùng tự nhập):\n"
                final_symptom_prompt += f"{other_symptoms}\n"

            # Kiểm tra xem người dùng có nhập gì không
            if not checked_symptoms and not other_symptoms:
                st.warning("Vui lòng chọn ít nhất một triệu chứng hoặc mô tả thêm.")
            else:
                with st.spinner("Đang gửi yêu cầu tới Gemini..."):
                    analysis_result = call_gemini(api_key, final_symptom_prompt)
                    
                    if analysis_result.startswith("LỖI:"):
                        st.error(analysis_result)
                    else:
                        st.subheader("Kết quả Phân tích từ Gemini")
                        st.markdown(analysis_result) # Dùng markdown để hiển thị format