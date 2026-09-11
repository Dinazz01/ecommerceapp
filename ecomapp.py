
import streamlit as st
import pandas as pd
# import numpy as np
import pickle


# ============================================      
# Load trained model
# ============================================

with open("ecom.pkl", "rb") as file:
    model = pickle.load(file)

feature_columns = model.feature_names_in_.tolist()

print("Model loaded successfully")
print("Number of features:", len(feature_columns))
print(feature_columns)

#============================================
# Page configuration
# ============================================
st.set_page_config(
    page_title="E-commerce Returned Product Predictor",
    page_icon="ecom.png",
    layout="centered"
)
st.title("E-commerce Returned Product Predictor")

st.write("This application predicts whether a product will be returned or not based on the input features provided by the user.")

# from PIL import Image
# import io
# img = Image.open("ecom.png")
# img_bytes = io.BytesIO()
# img.save(img_bytes, format='PNG')
# img_bytes = img_bytes.getvalue()    
# st.image(img_bytes)

st.image(
    "ecom.png",
    caption="E-Commerce Return Prediction"
)



# ---------------------------------------
# User inputs
# ---------------------------------------

customer_age = st.number_input(
    "Customer Age",
    min_value=18,
    max_value=100,
    value=35
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

product_category = st.selectbox(
    "Product Category",
    [
        "Electronics",
        "Clothing",
        "Home & Kitchen",
        "Beauty",
        "Sports",
        "Books"
    ]
)

product_price = st.number_input(
    "Product Price ($)",
    min_value=0.0,
    value=100.0
)

quantity = st.number_input(
    "Quantity",
    min_value=1,
    max_value=20,
    value=1
)

discount_percent = st.slider(
    "Discount Percent",
    min_value=0,
    max_value=50,
    value=10
)

shipping_cost = st.number_input(
    "Shipping Cost ($)",
    min_value=0.0,
    value=8.0
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Credit Card",
        "Debit Card",
        "PayPal",
        "Apple Pay",
        "Google Pay"
    ]
)

device_type = st.selectbox(
    "Device Type",
    [
        "Mobile",
        "Desktop",
        "Tablet"
    ]
)

customer_rating = st.slider(
    "Customer Rating",
    min_value=1,
    max_value=5,
    value=4
)

delivery_days = st.number_input(
    "Delivery Days",
    min_value=1,
    max_value=30,
    value=5
)

previous_orders = st.number_input(
    "Previous Orders",
    min_value=0,
    value=5
)

#---------------------------------------
# feature engineering
#---------------------------------------    
def create_features(data):

    # Gross order value
    data["Gross_Order_Value"] = (
        data["Product_Price"] *
        data["Quantity"]
    )

    # Discount amount
    data["Discount_Amount"] = (
        data["Gross_Order_Value"] *
        data["Discount_Percent"] /
        100
    )

    # Net product value
    data["Net_Product_Value"] = (
        data["Gross_Order_Value"] -
        data["Discount_Amount"]
    )

    # Total amount
    data["Total_Amount"] = (
        data["Net_Product_Value"] +
        data["Shipping_Cost"]
    )

    # Shipping percentage
    data["Shipping_Cost_Percent"] = (
        data["Shipping_Cost"] /
        data["Gross_Order_Value"] *
        100
    )

    # Age group
    data["Age_Group"] = pd.cut(
        data["Customer_Age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "65+"
        ]
    )

    # Delivery speed
    data["Delivery_Speed"] = pd.cut(
        data["Delivery_Days"],
        bins=[0, 3, 5, 7, float("inf")],
        labels=[
            "Fast",
            "Normal",
            "Slow",
            "Very Slow"
        ]
    )

    # Customer type
    data["Customer_Type"] = pd.cut(
        data["Previous_Orders"],
        bins=[-1, 2, 5, 10, float("inf")],
        labels=[
            "New",
            "Occasional",
            "Regular",
            "Loyal"
        ]
    )

    # Low rating
    data["Low_Rating"] = (
        data["Customer_Rating"] <= 2
    ).astype(int)

    # High discount
    data["High_Discount"] = (
        data["Discount_Percent"] >= 25
    ).astype(int)

    return data

#---------------------------------------
# build the prediction button
#---------------------------------------
if st.button("Predict Return"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        "Customer_Age": [customer_age],
        "Gender": [gender],
        "Product_Category": [product_category],
        "Product_Price": [product_price],
        "Quantity": [quantity],
        "Discount_Percent": [discount_percent],
        "Shipping_Cost": [shipping_cost],
        "Payment_Method": [payment_method],
        "Device_Type": [device_type],
        "Customer_Rating": [customer_rating],
        "Delivery_Days": [delivery_days],
        "Previous_Orders": [previous_orders]
    })

    # Apply your feature engineering
    input_data = create_features(input_data)

    # One-hot encode categorical columns
    input_encoded = pd.get_dummies(
       input_data,
       dtype=int
    )

    # Make input columns exactly match training columns
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict
    prediction = model.predict(input_encoded)[0]

    probability = model.predict_proba(input_encoded)[0][1]

    # 6. Display prediction


    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ This order is likely to be returned."
        )

    else:

        st.success(
            "✅ This order is unlikely to be returned."
        )

    # -----------------------------------
    # 7. Display probability
    # -----------------------------------

    st.metric(
        label="Probability of Return",
        value=f"{probability * 100:.1f}%"
    )


 
