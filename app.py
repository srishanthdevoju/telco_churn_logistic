# =========================
# app.py
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# ------------------ LOAD CSS ------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    return df

df = load_data()

st.title("📊 Telco Customer Churn Prediction (Logistic Regression)")

# ------------------ DATA OVERVIEW ------------------
st.header("Dataset Overview")
st.write(df.head())
st.write("Shape:", df.shape)

# ------------------ DATA PREPROCESSING ------------------
st.header("Data Preprocessing")

df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

st.write("Processed Data Sample")
st.write(df.head())

# ------------------ SPLIT DATA ------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ MODEL TRAINING ------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ------------------ CONFUSION MATRIX ------------------
st.header("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# ------------------ CLASSIFICATION REPORT ------------------
st.header("Classification Report")
st.text(classification_report(y_test, y_pred))

# ------------------ ROC CURVE ------------------
st.header("ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0, 1], [0, 1], linestyle="--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()

st.pyplot(fig2)

# ------------------ DATA ANALYSIS ------------------
st.header("Dataset Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Distribution")
    fig3, ax3 = plt.subplots()
    df["Churn"].value_counts().plot(kind="bar", ax=ax3)
    st.pyplot(fig3)

with col2:
    st.subheader("Correlation with Churn")
    corr = df.corr()["Churn"].sort_values(ascending=False)
    st.write(corr)

# ------------------ USER INPUT PREDICTION ------------------
st.header("Predict Churn for New Customer")

user_data = []
for col in X.columns:
    val = st.number_input(col, float(df[col].min()), float(df[col].max()))
    user_data.append(val)

if st.button("Predict"):
    user_data = scaler.transform([user_data])
    prediction = model.predict(user_data)[0]
    st.success("Churn" if prediction == 1 else "Not Churn")
