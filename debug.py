import streamlit as st
import pandas as pd
import pickle
import os
import sys

st.set_page_config(page_title="Debug Model", page_icon="⚡")

st.title('Debug Model Pickle')

# Tampilkan informasi environment
st.subheader("Environment Info")
st.write(f"Python version: {sys.version}")
st.write(f"Working directory: {os.getcwd()}")
st.write(f"Files in directory: {os.listdir('.')}")

# Cek file model
model_path = 'linear_regression_model.pkl'
if os.path.exists(model_path):
    st.success(f"✅ Model file found: {model_path}")
    file_size = os.path.getsize(model_path)
    st.write(f"File size: {file_size} bytes")
else:
    st.error(f"❌ Model file NOT found: {model_path}")
    st.stop()

# Coba load model dengan detail error
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
    st.write(f"Model type: {type(model)}")
    st.write(f"Model attributes: {dir(model)}")
    
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    
    # Coba cara lain
    st.subheader("Alternative loading methods:")
    
    # Method 2: Coba dengan encoding
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file, encoding='latin1')
        st.success("✅ Model loaded with latin1 encoding!")
    except Exception as e2:
        st.error(f"❌ Latin1 encoding failed: {e2}")
    
    # Method 3: Coba dengan bytes
    try:
        with open(model_path, 'rb') as file:
            model_bytes = file.read()
            model = pickle.loads(model_bytes)
        st.success("✅ Model loaded with pickle.loads!")
    except Exception as e3:
        st.error(f"❌ Pickle loads failed: {e3}")
