import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# 📌 Load the Dataset
df = pd.read_csv("AmazonDT_Dataset1.csv")

# 📌 Streamlit UI
st.title("📦 Amazon Delivery Time Prediction")

# 📌 Exploratory Data Analysis (EDA)
st.header("1️⃣ Exploratory Data Analysis")

st.subheader("📊 Dataset Overview")
st.write(df.head())

st.subheader("📌 Basic Statistics")
st.write(df.describe())

st.subheader("🔍 Missing Values")
st.write(df.isnull().sum())

# 📌 Handling Missing Values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 📌 Outlier Detection & Removal
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 📌 Convert DateTime Columns
if 'Order_Time' in df.columns:
    df['Order_Time'] = pd.to_datetime(df['Order_Time'], format="%H:%M:%S", errors='coerce')
    df['order_hour'] = df['Order_Time'].dt.hour
    df['order_minute'] = df['Order_Time'].dt.minute

# 📌 Encoding Categorical Variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 📌 Feature Engineering
if 'Delivery_Time' in df.columns:
    X = df.drop(columns=['Delivery_Time'])
    y = df['Delivery_Time']
else:
    st.error("⚠️ 'Delivery_Time' column is missing!")
    st.stop()

# 📌 Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 📌 Model Training
st.header("2️⃣ Model Training & Evaluation")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("🔢 Model Performance Metrics")
st.write(f"✅ *Mean Absolute Error (MAE):* {mae}")
st.write(f"✅ *Mean Squared Error (MSE):* {mse}")
st.write(f"✅ *R² Score:* {r2}")

# 📌 Save Model, Scaler, and Encoders
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

st.success("✅ Model and encoders saved successfully!")

# 📌 Prediction Section
st.header("3️⃣ Predict Delivery Time")
st.write("📥 Enter order details below:")

features = {}
for col in X.columns:
    if col in label_encoders:
        features[col] = st.selectbox(f"Select {col}", options=label_encoders[col].classes_)
    else:
        features[col] = st.number_input(f"Enter {col}")

if st.button("🚀 Predict Delivery Time"):
    input_df = pd.DataFrame([features])

    # Apply Encoding
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Apply Scaling
    input_df_scaled = scaler.transform(input_df)

    # Predict
    predicted_time = model.predict(input_df_scaled)
    st.success(f"🚚 Estimated Delivery Time: {predicted_time[0]:.2f} minutes")