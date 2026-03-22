import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime




# Page config
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ FIX: Initialize session state FIRST (right after set_page_config)
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.churn_prob = None
    st.session_state.segment = None
    st.session_state.prediction_history = []



# Custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }

    /* Card styling */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 25px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }

    /* Risk indicators */
    .high-risk {
        background: #ff4757;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .medium-risk {
        background: #ffa502;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .low-risk {
        background: #26de81;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff4757, #ffa502, #26de81);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# Load models
@st.cache_resource
def load_models():
    try:
        churn_model = joblib.load('churn_model.pkl')
        kmeans_model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        training_columns = joblib.load('training_columns.pkl')
        return churn_model, kmeans_model, scaler, training_columns
    except:
        return None, None, None, None


# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.churn_prob = None
    st.session_state.segment = None
    st.session_state.prediction_history = []

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("# 🛡️")
with col2:
    st.markdown("# ChurnGuard AI")
    st.markdown("### Intelligent Customer Retention System")

st.markdown("---")

# Load models
churn_model, kmeans_model, scaler, training_columns = load_models()

if churn_model is None:
    st.error("⚠️ Models not found. Please train and save your models first.")
    st.stop()

# ==================== ORIGINAL SIDEBAR ====================
st.sidebar.header("📝 Enter Customer Details")

# Numerical inputs
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 20.0, 120.0, 70.0)
total_charges = monthly_charges * tenure

# Categorical inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
senior = st.sidebar.selectbox("Senior Citizen?", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method",
                                      ["Electronic check", "Mailed check",
                                       "Bank transfer (automatic)", "Credit card (automatic)"])

# Service checkboxes
st.sidebar.subheader("🛠️ Additional Services")
online_security = st.sidebar.checkbox("Online Security")
online_backup = st.sidebar.checkbox("Online Backup")
device_protection = st.sidebar.checkbox("Device Protection")
tech_support = st.sidebar.checkbox("Tech Support")
streaming_tv = st.sidebar.checkbox("Streaming TV")
streaming_movies = st.sidebar.checkbox("Streaming Movies")

# Calculate total services (defined here for use everywhere)
service_list = [online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies]
total_services = sum(service_list) + (1 if phone_service == "Yes" else 0)
if internet_service != "No":
    total_services += 1

# Predict button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 PREDICT CHURN", type="primary", use_container_width=True)

# Create tenure group
if tenure <= 12:
    tenure_group = 0
elif tenure <= 24:
    tenure_group = 1
elif tenure <= 48:
    tenure_group = 2
else:
    tenure_group = 3

# Prepare input data
input_data = pd.DataFrame([{
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'total_services': total_services,
    'tenure_group': tenure_group,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner_Yes': 1 if partner == "Yes" else 0,
    'Dependents_Yes': 1 if dependents == "Yes" else 0,
    'PhoneService_Yes': 1 if phone_service == "Yes" else 0,
    'OnlineSecurity_Yes': 1 if online_security else 0,
    'OnlineBackup_Yes': 1 if online_backup else 0,
    'DeviceProtection_Yes': 1 if device_protection else 0,
    'TechSupport_Yes': 1 if tech_support else 0,
    'StreamingTV_Yes': 1 if streaming_tv else 0,
    'StreamingMovies_Yes': 1 if streaming_movies else 0,
    'gender_Male': 1 if gender == "Male" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
}])

# ==================== MODERN MAIN UI ====================
if predict_button:
    # Reindex to match training columns
    input_data = input_data.reindex(columns=training_columns, fill_value=0)

    # Scale numerical features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'total_services']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Make predictions
    st.session_state.churn_prob = churn_model.predict_proba(input_data)[0][1]
    st.session_state.segment = kmeans_model.predict(
        input_data[['tenure', 'MonthlyCharges', 'total_services', 'tenure_group']]
    )[0]
    st.session_state.prediction_made = True

    # Add to history
    st.session_state.prediction_history.append({
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'probability': st.session_state.churn_prob,
        'segment': st.session_state.segment
    })

# Display results if prediction exists
if st.session_state.prediction_made:
    # Welcome card
    st.markdown("""
    <div class="card">
        <h2>📊 Analysis Results</h2>
        <p>Based on our AI model analysis, here's the customer profile and recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #666; font-size: 16px;'>Churn Probability</h3>
        """, unsafe_allow_html=True)
        prob = st.session_state.churn_prob
        st.markdown(f"<h2 style='color: #333;'>{prob:.1%}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #666; font-size: 16px;'>Risk Level</h3>
        """, unsafe_allow_html=True)
        if prob >= 0.45:
            st.markdown("<p style='color: #ff4757; font-weight: bold; font-size: 18px;'>⚠️ HIGH RISK</p>",
                        unsafe_allow_html=True)
        elif prob >= 0.3:
            st.markdown("<p style='color: #ffa502; font-weight: bold; font-size: 18px;'>⚠️ MEDIUM RISK</p>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #26de81; font-weight: bold; font-size: 18px;'>✅ LOW RISK</p>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #666; font-size: 16px;'>Customer Segment</h3>
        """, unsafe_allow_html=True)
        segment_names = {
            0: "🏆 Loyal High-Value",
            1: "🌱 New/Low-Engagement",
            2: "📈 Growing/Regular",
            3: "💤 Low-Usage/Loyal"
        }
        st.markdown(
            f"<p style='color: #667eea; font-weight: bold; font-size: 18px;'>{segment_names[st.session_state.segment]}</p>",
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #666; font-size: 16px;'>Est. Lifetime Value</h3>
        """, unsafe_allow_html=True)
        clv_multiplier = [24, 12, 18, 30]
        clv = monthly_charges * clv_multiplier[st.session_state.segment]
        st.markdown(f"<h2 style='color: #333;'>${clv:.0f}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Risk Analysis")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 30], 'color': "#26de81"},
                    {'range': [30, 45], 'color': "#ffa502"},
                    {'range': [45, 100], 'color': "#ff4757"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 45
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Recommended Actions")

        segment = st.session_state.segment
        prob = st.session_state.churn_prob

        if segment == 0:
            if prob >= 0.45:
                st.info("🎁 **Urgent:** Offer exclusive loyalty reward + personal manager call")
            else:
                st.success("⭐ **Retain:** Upsell premium services, invite to VIP program")
        elif segment == 1:
            if prob >= 0.45:
                st.warning("📧 **Intervene:** Send welcome offer + onboarding call")
            else:
                st.info("📱 **Engage:** Educational content about service benefits")
        elif segment == 2:
            if prob >= 0.45:
                st.warning("🆓 **Retain:** Offer 3 months free upgrade")
            else:
                st.info("📈 **Grow:** Suggest bundle discounts")
        else:
            if prob >= 0.45:
                st.error("🔍 **Investigate:** Check service quality, offer tech support")
            else:
                st.success("💬 **Maintain:** Regular satisfaction surveys")

    # Customer Profile Summary
    st.markdown("### 📋 Customer Profile Summary")
    profile_df = pd.DataFrame({
        'Attribute': ['Tenure', 'Monthly Charges', 'Contract', 'Payment Method', 'Total Services', 'Internet Service'],
        'Value': [f"{tenure} months", f"${monthly_charges}", contract, payment_method, total_services, internet_service]
    })
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    # Segment Insights
    st.markdown("### 📊 Segment Insights")
    col1, col2 = st.columns(2)

    with col1:
        segment_stats = pd.DataFrame({
            'Segment': ['Loyal High-Value', 'New/Low-Engagement', 'Growing/Regular', 'Low-Usage/Loyal'],
            'Churn Rate': ['12%', '46%', '31%', '4%'],
            'Avg CLV': ['$1,440', '$720', '$1,080', '$1,800'],
            'Strategy': ['Retention rewards', 'Win-back emails', 'Upgrade offers', 'Cross-sell']
        })
        st.dataframe(segment_stats, use_container_width=True, hide_index=True)

    with col2:
        # Pie chart of segment distribution
        segment_counts = [35, 25, 25, 15]
        fig = px.pie(values=segment_counts,
                     names=['Loyal High-Value', 'New/Low-Engagement', 'Growing/Regular', 'Low-Usage/Loyal'],
                     title="Segment Distribution",
                     color_discrete_sequence=['#667eea', '#764ba2', '#ffa502', '#26de81'])
        st.plotly_chart(fig, use_container_width=True)

    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("### 📜 Recent Predictions")
        history_df = pd.DataFrame(st.session_state.prediction_history[-5:])
        history_df['probability'] = history_df['probability'].apply(lambda x: f"{x:.1%}")
        history_df['segment'] = history_df['segment'].map(segment_names)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

else:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h1 style='font-size: 80px; margin: 0;'>🛡️</h1>
            <h2 style='color: #666;'>Welcome to ChurnGuard AI</h2>
            <p style='color: #999; font-size: 18px;'>Enter customer details in the sidebar and click PREDICT CHURN to get started</p>
        </div>
        """, unsafe_allow_html=True)

        # Features
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <h3 style='color: #667eea; font-size: 40px; margin: 0;'>🔮</h3>
                <h4>Predict Churn</h4>
                <p style='color: #999;'>AI-powered churn prediction</p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <h3 style='color: #667eea; font-size: 40px; margin: 0;'>👥</h3>
                <h4>Segment Customers</h4>
                <p style='color: #999;'>Automatic customer segmentation</p>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <h3 style='color: #667eea; font-size: 40px; margin: 0;'>💰</h3>
                <h4>Calculate CLV</h4>
                <p style='color: #999;'>Customer lifetime value estimation</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; color: #999;'>
        <p>ChurnGuard AI v1.0 | Powered by Machine Learning | © 2024</p>
    </div>
    """, unsafe_allow_html=True)