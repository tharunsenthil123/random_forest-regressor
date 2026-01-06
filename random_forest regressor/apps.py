import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Ames Housing Price Prediction", layout="centered")

st.title("🏠 Ames Housing Price Prediction")
st.write("Random Forest Regressor")

# -----------------------------
# 1️⃣ Widget MUST be outside cache
# -----------------------------
uploaded_file = st.file_uploader("Upload AmesHousing.csv", type=["csv"])

if uploaded_file is None:
    st.info("Please upload AmesHousing.csv to continue")
    st.stop()

# -----------------------------
# 2️⃣ Cache ONLY pure function
# -----------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Features & Target
# -----------------------------
FEATURES = [
    "Overall Qual",
    "Gr Liv Area",
    "Garage Cars",
    "Total Bsmt SF",
    "Year Built"
]
TARGET = "SalePrice"

X = df[FEATURES]
y = df[TARGET]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter House Details")

overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
gr_liv_area = st.number_input("Ground Living Area (sq ft)", value=1500)
garage_cars = st.slider("Garage Cars", 0, 4, 2)
total_bsmt = st.number_input("Total Basement Area (sq ft)", value=800)
year_built = st.number_input("Year Built", value=2000)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict House Price"):
    input_data = np.array([[overall_qual, gr_liv_area, garage_cars, total_bsmt, year_built]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🏷️ Estimated House Price: ${prediction:,.0f}")
