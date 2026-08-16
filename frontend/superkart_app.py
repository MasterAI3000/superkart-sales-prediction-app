
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
def load_model():
    # Replace with your actual model filename
    return joblib.load("superkart_sales_model_v1_0.joblib")

model = load_model()

# Streamlit UI for Superkart Sales Prediction
st.title("Superkart Sales Prediction App")
st.write("""
This app predicts the **Total Store Sales** for a particular product based on its attributes and the store's profile.
Kindly enter the details below to estimate revenue.
""")

# --- Collect User Input ---
st.subheader("Product Details")
col1, col2 = st.columns(2)

with col1:
    Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.0)
    Product_MRP = st.number_input("Maximum Retail Price (MRP)", min_value=0.0, value=150.0)
    Product_Sugar_Content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])

with col2:
    Product_Allocated_Area = st.number_input("Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.05, format="%.4f")
    Product_Type = st.selectbox("Product Category", [
        "Meat", "Snack Foods", "Hard Drinks", "Dairy", "Canned", "Soft Drinks",
        "Health and Hygiene", "Baking Goods", "Breads", "Breakfast", "Frozen Foods",
        "Fruits and Vegetables", "Household", "Seafood", "Starchy Foods", "Others"
    ])

st.subheader("Store Details")
col3, col4 = st.columns(2)

with col3:
    Store_Size = st.selectbox("Store Size", ["High", "Medium", "Small"])
    Store_Establishment_Year = st.number_input("Year Established", min_value=1980, max_value=2026, value=2000)

with col4:
    Store_Location_City_Type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
    Store_Type = st.selectbox("Store Category", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])

# --- Preprocessing Input ---
# Create input dataframe matching the model's expected features
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Type': Product_Type,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

# Note: If your model was trained on log-transformed variables or one-hot encoded columns,
# you would apply those transformations here to input_data before calling model.predict().

# --- Predict Button ---
if st.button("Predict Sales"):
    # Perform prediction
    prediction = model.predict(input_data)[0]

    # Handle log transformation if applied during training
    # if prediction was trained on log(sales), use: final_sales = np.expm1(prediction)
    final_sales = max(0, prediction) # Ensure we don't show negative sales

    st.success(f"### Estimated Total Sales: {final_sales:,.2f} USD")
    st.info("This estimate is based on the historical patterns of product performance across similar store types.")
