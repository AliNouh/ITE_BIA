import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.compare_methods import run_all

st.set_page_config(page_title='GA Feature Selection - BIA601', layout='wide')
st.title("اختيار الميزات باستخدام خوارزمية وراثية — BIA601 (RAFD)")

uploaded = st.file_uploader("ارفع ملف CSV (يجب أن يحتوي على عمود target)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("معاينة البيانات:")
    st.write(df.head())
    target_col = st.selectbox("اختر عمود الهدف (target)", df.columns, index=len(df.columns)-1)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
else:
    st.info("لم يتم رفع ملف. سيتم استخدام بيانات Breast Cancer الافتراضية.")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data
    y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("عدد الميزات المرجو المقارنة به (k)", min_value=1, max_value=max(1, X.shape[1]//2), value=max(1, X.shape[1]//4))
if st.button("تشغيل التحليل"):
    with st.spinner("جارٍ تشغيل المقارنات..."):
        results = run_all(X_scaled, y, k_features=k, verbose=False)
    st.success("اكتمل التحليل")
    st.write("نتائج الملخّص:")
    rows = []
    for method, metrics in results.items():
        rows.append({'الطريقة': method, 'الدقة (accuracy)': metrics['accuracy'], 'F1 (f1_macro)': metrics['f1'], 'عدد الميزات': metrics['n_features']})
    st.table(pd.DataFrame(rows))
    st.write("يمكن تنزيل النتائج JSON:")
    st.download_button("تحميل النتائج (JSON)", data=pd.io.json.dumps(results, indent=2), file_name="results.json")
