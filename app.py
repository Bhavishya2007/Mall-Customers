import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Custom CSS for cute UI
st.markdown("""
<style>
    .stApp {
        background-color: #fff5f8;
    }
    .stTitle {
        color: #e84393;
        font-family: 'Comic Sans MS', cursive;
        text-align: center;
    }
    .stMarkdown h3 {
        color: #000000;
        font-weight: bold;
    }
    .stMarkdown h4 {
        color: #000000;
        font-weight: bold;
    }
    .stNumberInput label {
        color: #000000 !important;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(45deg, #fd79a8, #e84393);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 30px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #e84393, #fd79a8);
    }
    div[data-testid="stSuccess"] {
        background-color: #dfe6e9;
        border-radius: 15px;
        padding: 15px;
    }
    div[data-testid="stInfo"] {
        background-color: #ffeaa7;
        border-radius: 15px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# App title with cute emoji
st.markdown("<h1 style='text-align: center; color: #e84393;'>🌸 Mall Customer Clustering 🌸</h1>", unsafe_allow_html=True)

# Input fields in cute boxes
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💰 Annual Income (k$)")
    annual_income = st.number_input("", min_value=0, max_value=150, value=60, placeholder="Enter income", label_visibility="collapsed")

with col2:
    st.markdown("#### 💳 Spending Score (1-100)")
    spending_score = st.number_input("", min_value=1, max_value=100, value=50, placeholder="Enter score", label_visibility="collapsed")

st.write("")

# Prediction button
if st.button("🔮 Predict My Segment"):
    # Predict cluster
    input_data = np.array([[annual_income, spending_score]])
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)[0]
    
    # Cluster descriptions (professional with money symbols)
    cluster_info = {
        0: "💰 Low Income + Low Spending | Careful Spenders",
        1: "💎 High Income + High Spending | Target Customers",
        2: "🏦 High Income + Low Spending | Conservative Spenders",
        3: "🛒 Low Income + High Spending | Careless Spenders",
        4: "⚖️ Medium Income + Medium Spending | Average Customers"
    }
    
    # Output
    st.success(f"📊 Customer Segment: **Cluster {cluster}**")
    st.info(cluster_info.get(cluster, 'Unknown'))

# Footer
st.markdown("---")


