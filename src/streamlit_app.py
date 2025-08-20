import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Set the page configuration for a wider layout
st.set_page_config(layout="wide")

# --- Custom CSS for Styling (Updated Dark Mode & Neon Headings) ---
def apply_custom_css():
    """
    Applies custom CSS for a dark theme and neon text effect, with
    improved label visibility and reduced contrast.
    """
    st.markdown("""
    <style>
    /* Main body styling for a softer dark mode */
    .stApp {
        background-color: #1a202c; /* Softer dark grey */
        color: #FFFFFF; /* Light text color */
    }
    /* Set the sidebar background */
    [data-testid="stSidebar"] {
        background-color: #2d3748; /* Slightly darker than main body */
        color: #FFFFFF; /* #e2e8f0 */
    }
    
    /* Neon Text effect for titles */
    .neon-title {
        color: #39FF14; /* Neon green */
        text-shadow:
            0 0 5px #39FF14,
            0 0 10px #39FF14,
            0 0 20px #39FF14,
            0 0 40px #39FF14,
            0 0 80px #39FF14;
    }
    .neon-header {
        color: #00FFFF; /* Neon cyan */
        text-shadow:
            0 0 5px #00FFFF,
            0 0 10px #00FFFF,
            0 0 20px #00FFFF;
    }
    
    /* Fix for all input widget label colors in the main content area */
    div[data-testid="stForm"] label p, div[data-testid="stVerticalBlock"] label p {
        color: #FFFFFF !important;
    }

    /* Fix for sidebar label colors */
    [data-testid="stSidebar"] .st-emotion-cache-1wv9397 p, [data-testid="stSidebar"] .st-emotion-cache-1629p8f p {
        color: #FFFFFF !important;
    }

    /* Fix for st.metric labels and values */
    .st-emotion-cache-s8z6q p, .st-emotion-cache-1wv9397 p, .st-emotion-cache-121p54e {
        color: #FFFFFF !important;
    }

    /* Fix for st.metric delta */
    .st-emotion-cache-13l37u7 {
        color: #FFFFFF !important;
    }

    /* Style for expander titles */
    .streamlit-expanderHeader {
        font-size: 1.25rem;
        font-weight: bold;
    }
    
    /* Style for buttons */
    .stButton>button {
        color: #00FFFF;
        background-color: #2d3748;
        border: 2px solid #00FFFF;
        border-radius: 10px;
        box-shadow: 0 0 10px #00FFFF;
    }
    .stButton>button:hover {
        background-color: #00FFFF;
        color: #2d3748;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Apply the custom CSS at the start
apply_custom_css()

# --- Data Loading and Model Training (Cached) ---
@st.cache_resource
def load_data_and_train_model():
    """
    Loads the dataset and trains a logistic regression model.
    """
    try:
        df = pd.read_csv(r'src/churn_dataset1.csv')
    except FileNotFoundError:
        st.error("Dataset 'churn_dataset1.csv' not found. Please ensure it's in the same directory.")
        return None, None

    # Clean and preprocess the data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    categorical_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)

    return model, df

model, df = load_data_and_train_model()

# --- Functions for Each Sidebar Page ---

def show_prediction_page(model, df):
    """
    Creates the customer churn prediction page.
    """
    st.markdown("<h1 class='neon-title'>📞 Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background-color:#2d3748; padding: 15px; border-radius: 10px;">
        <p style="font-size: 18px;">
        Fill out the customer information below to predict the likelihood of churn.
        </p>
        </div>
        <br>
    """, unsafe_allow_html=True)

    with st.container():
        # User Input Form
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
            gender = st.selectbox("Gender", df['gender'].unique())
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ['Yes', 'No'])
            dependents = st.selectbox("Dependents", ['Yes', 'No'])
            
        with col2:
            phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
            multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
            internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
            online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
            device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
            tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
            streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
            streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
            contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
            payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        
    st.markdown("---")
    
    if st.button("Predict Churn"):
        # Create a DataFrame from the user inputs
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }])
        
        # Make a prediction
        prediction_proba = model.predict_proba(input_data)[0]
        churn_proba = prediction_proba[1]
        
        st.markdown("<h3 class='neon-header'>Prediction Result</h3>", unsafe_allow_html=True)
        
        if churn_proba > 0.5:
            st.error(f"The model predicts this customer **is likely to churn**.")
            st.warning(f"Probability of Churn: **{churn_proba:.2%}**")
        else:
            st.success(f"The model predicts this customer is **not likely to churn**.")
            st.info(f"Probability of Not Churning: **{(1 - churn_proba):.2%}**")

def show_dashboard_page(df):
    """
    Creates the dashboards page with KPIs.
    """
    st.markdown("<h1 class='neon-title'>Churn Dashboard & KPIs</h1>", unsafe_allow_html=True)
    st.markdown("A quick overview of key metrics from the dataset.")

    # Convert 'Churn' to numeric for calculations
    df['Churn_num'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Calculate KPIs
    total_customers = df.shape[0]
    churned_customers = df[df['Churn_num'] == 1].shape[0]
    churn_rate = (churned_customers / total_customers) * 100
    avg_monthly_charges = df['MonthlyCharges'].mean()
    avg_tenure_churned = df[df['Churn'] == 'Yes']['tenure'].mean()
    
    st.markdown("---")
    
    # Display KPIs in a column layout for better presentation
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<h4 class='neon-header'>Total Customers</h4>", unsafe_allow_html=True)
        st.metric(label="Total Customers", value=f"{total_customers:,}")
    with col2:
        st.markdown("<h4 class='neon-header'>Churn Rate</h4>", unsafe_allow_html=True)
        st.metric(label="Overall Churn Rate", value=f"{churn_rate:.2f}%")
    with col3:
        st.markdown("<h4 class='neon-header'>Avg. Monthly Charges</h4>", unsafe_allow_html=True)
        st.metric(label="Average Monthly Charges", value=f"${avg_monthly_charges:.2f}")
    with col4:
        st.markdown("<h4 class='neon-header'>Avg. Tenure of Churned Customers</h4>", unsafe_allow_html=True)
        st.metric(label="Avg. Tenure (months)", value=f"{avg_tenure_churned:.2f}")

    st.markdown("---")
    st.markdown("<h3 class='neon-header'>Data Table</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(10))

def show_visualizations_page(df):
    """
    Creates the visualizations page with various plots.
    """
    st.markdown("<h1 class='neon-title'>Visualizations</h1>", unsafe_allow_html=True)
    st.markdown("Explore the data through different charts and plots.")

    # Convert 'Churn' to numeric for some plots
    df['Churn_num'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # --- PIE CHART ---
    st.markdown("<h3 class='neon-header'>1. Churn Distribution (Pie Chart)</h3>", unsafe_allow_html=True)
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn Status', 'Count']
    fig_pie = px.pie(churn_counts, values='Count', names='Churn Status', title='Churn vs. Non-Churn',
                     color_discrete_sequence=['#39FF14', '#FF5733'])
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")

    # --- BAR PLOT ---
    st.markdown("<h3 class='neon-header'>2. Churn by Contract Type (Bar Plot)</h3>", unsafe_allow_html=True)
    contract_churn = df.groupby('Contract')['Churn_num'].mean().reset_index()
    fig_bar = px.bar(contract_churn, x='Contract', y='Churn_num', color='Contract',
                     title='Churn Rate by Contract Type',
                     labels={'Churn_num': 'Churn Rate', 'Contract': 'Contract'},
                     color_discrete_map={'Month-to-month': '#FF5733', 'One year': '#FFC300', 'Two year': '#39FF14'})
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # --- HISTOGRAM ---
    st.markdown("<h3 class='neon-header'>3. Customer Tenure Distribution (Histogram)</h3>", unsafe_allow_html=True)
    fig_hist = px.histogram(df, x='tenure', color='Churn', title='Tenure Distribution by Churn Status',
                            color_discrete_map={'Yes': '#FF5733', 'No': '#39FF14'})
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")

    # --- HEATMAP ---
    st.markdown("<h3 class='neon-header'>4. Correlation Heatmap</h3>", unsafe_allow_html=True)
    
    # Select and convert relevant features for correlation
    numeric_df = df.copy()
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
        if col in numeric_df.columns:
            numeric_df[col] = numeric_df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # Calculate correlation matrix
    corr_matrix = numeric_df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']].corr()

    # Create a Plotly heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        hoverinfo='z+x+y',
    ))

    fig_heatmap.update_layout(
        title='Correlation Matrix of Key Features',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed',
        height=600,
        margin=dict(l=100, r=20, t=50, b=100)
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

def show_insights_page(df):
    """
    Creates the insights page with key findings from the data.
    """
    st.markdown("<h1 class='neon-title'>Key Insights</h1>", unsafe_allow_html=True)
    st.markdown("Here are some key findings from the dataset that can help inform business decisions.")
    st.markdown("---")

    st.markdown("<h3 class='neon-header'>Top Factors Influencing Churn</h3>", unsafe_allow_html=True)
    st.markdown("""
    * **Contract Type:** Customers on **month-to-month** contracts have a significantly higher churn rate than those on one or two-year contracts. This is a primary driver of churn.
    * **Internet Service:** Customers with **Fiber Optic** internet service show a higher churn rate compared to those with DSL. This could be due to service quality issues or a highly competitive market for fiber providers.
    * **Electronic Check:** Customers who use **Electronic Check** as their payment method churn at a much higher rate. This may indicate dissatisfaction and a lack of commitment to the service.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 class='neon-header'>Customer Profile Insights</h3>", unsafe_allow_html=True)
    st.markdown("""
    * **Tenure:** The churn rate is highest for customers with a short **tenure** (less than 12 months). Customer retention efforts should focus on this group.
    * **Optional Services:** Services like **Online Security** and **Tech Support** are effective for retention. Customers without these services are more likely to churn.
    """)

# --- Main App Logic ---

# Sidebar navigation
st.sidebar.markdown("<h2 class='neon-header'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Prediction", "Dashboards", "Visualizations", "Insights"])

if model is None:
    st.stop()

# Display the selected page
if page == "Prediction":
    show_prediction_page(model, df)
elif page == "Dashboards":
    show_dashboard_page(df)
elif page == "Visualizations":
    show_visualizations_page(df)
elif page == "Insights":
    show_insights_page(df)
