import streamlit as st
import pandas as pd
import numpy as np
import os

# Coba import joblib dulu, jika gagal gunakan pickle
try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    import pickle
    USE_JOBLIB = False

st.set_page_config(
    page_title="Prediksi Tagihan Listrik",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        # Coba beberapa format model
        model_files = [
            'linear_regression_model.joblib',
            'linear_regression_model.pkl',
            'model.joblib', 
            'model.pkl'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                if USE_JOBLIB and model_file.endswith('.joblib'):
                    model = joblib.load(model_file)
                    st.sidebar.success(f"✅ Model loaded with joblib: {model_file}")
                else:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    st.sidebar.success(f"✅ Model loaded with pickle: {model_file}")
                return model
        
        st.error("❌ No model file found!")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

# Demo prediction function jika model tidak ada
def demo_prediction(input_data):
    """Fallback prediction when model fails"""
    kwh = input_data['kwh'][0]
    ac_units = input_data['ac_units'][0]
    ac_hours = input_data['ac_hours_per_day'][0]
    family_size = input_data['family_size'][0]
    
    # Simple calculation
    base_cost = kwh * 1500
    ac_cost = ac_units * ac_hours * 500
    family_cost = family_size * 20000
    
    return base_cost + ac_cost + family_cost

# UI Components
st.title('⚡ Prediksi Tagihan Listrik Jakarta')
st.write('Aplikasi untuk memprediksi tagihan listrik berdasarkan parameter konsumsi.')

# Input form
st.sidebar.header('📊 Input Parameter')

kwh = st.sidebar.slider('Konsumsi KWH (kWh)', 150.0, 600.0, 350.0, 10.0)
ac_units = st.sidebar.slider('Jumlah AC', 0, 3, 1)
ac_hours_per_day = st.sidebar.slider('Jam AC per Hari', 0.0, 10.0, 5.0, 0.5)
family_size = st.sidebar.slider('Jumlah Anggota Keluarga', 2, 6, 4)
month_name = st.sidebar.selectbox('Bulan', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
tariff_class = st.sidebar.selectbox('Kelas Tarif', ['R1', 'R2', 'R3'])

# Prepare input data
input_data = pd.DataFrame({
    'kwh': [kwh],
    'ac_units': [ac_units],
    'ac_hours_per_day': [ac_hours_per_day],
    'family_size': [family_size],
    'month_name': [month_name],
    'tariff_class': [tariff_class]
})

# Show input data
st.subheader('📋 Parameter Input')
st.dataframe(input_data)

# Prediction
if st.sidebar.button('🎯 Prediksi Tagihan', type='primary'):
    if model is not None:
        try:
            # Prepare features for model prediction
            features = pd.DataFrame({
                'kwh': [kwh],
                'ac_units': [ac_units],
                'ac_hours_per_day': [ac_hours_per_day],
                'family_size': [family_size]
            })
            
            # Add month and tariff dummies
            for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                features[f'month_{month}'] = [1 if month_name == month else 0]
            
            for tariff in ['R1', 'R2', 'R3']:
                features[f'tariff_{tariff}'] = [1 if tariff_class == tariff else 0]
            
            prediction = model.predict(features)
            result = prediction[0]
            
        except Exception as e:
            st.warning(f"⚠️ Model prediction failed, using fallback: {e}")
            result = demo_prediction(input_data)
    else:
        st.warning("⚠️ Using demo prediction (model not available)")
        result = demo_prediction(input_data)
    
    # Display result
    st.subheader('📊 Hasil Prediksi')
    st.success(f"**Tagihan Listrik: Rp {result:,.2f}**")
    
    # Additional info
    with st.expander("💡 Informasi Tambahan"):
        st.write("""
        - Prediksi berdasarkan parameter input yang diberikan
        - Hasil dapat bervariasi tergantung kondisi aktual
        - Untuk perhitungan akurat, hubungi PLN setempat
        """)

# Footer
st.markdown("---")
st.markdown("© 2024 Aplikasi Prediksi Listrik | Built with Streamlit")
