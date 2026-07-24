import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Medical Cost Estimator", page_icon="🏥")

st.title("🏥 Medical Cost Estimation Using Machine Learning")
st.write("Predict medical insurance cost using Machine Learning Regression.")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# ---------------------------
# Encode Categorical Columns
# ---------------------------
label_encoders = {}

for column in ["sex", "smoker", "region"]:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    label_encoders[column] = encoder

# ---------------------------
# Split Data
# ---------------------------
X = data.drop("charges", axis=1)
y = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------
# Train Model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = r2_score(y_test, prediction)

st.success(f"Model Accuracy (R² Score): {accuracy:.2f}")

# ---------------------------
# User Input
# ---------------------------
st.header("Enter Patient Details")

age = st.slider("Age", 18, 100, 25)

sex = st.selectbox(
    "Gender",
    label_encoders["sex"].classes_
)

bmi = st.slider(
    "BMI",
    10.0,
    50.0,
    25.0
)

children = st.slider(
    "Children",
    0,
    5,
    0
)

smoker = st.selectbox(
    "Smoker",
    label_encoders["smoker"].classes_
)

region = st.selectbox(
    "Region",
    label_encoders["region"].classes_
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Medical Cost"):

    sex_encoded = label_encoders["sex"].transform([sex])[0]
    smoker_encoded = label_encoders["smoker"].transform([smoker])[0]
    region_encoded = label_encoders["region"].transform([region])[0]

    user_data = pd.DataFrame(
        [[
            age,
            sex_encoded,
            bmi,
            children,
            smoker_encoded,
            region_encoded
        ]],
        columns=X.columns
    )

    cost = model.predict(user_data)[0]

    st.subheader("Estimated Medical Cost")

    st.success(f"${cost:,.2f}")

# ---------------------------
# Feature Importance
# ---------------------------
st.header("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)