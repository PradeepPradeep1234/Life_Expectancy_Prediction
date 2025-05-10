# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Life Expectancy Predictor", layout="wide")

st.title("🌍 Life Expectancy Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/Life_Expectancy_Data.csv")
    df.columns = df.columns.str.strip()
    df.drop(columns=["Country"], inplace=True)
    le = LabelEncoder()
    df["Status"] = le.fit_transform(df["Status"])
    return df

df = load_data()

# EDA (show summary)
if st.checkbox("Show Data Summary"):
    st.write(df.describe())

# Correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Pairplot
if st.checkbox("Show Pairplot"):
    selected_cols = ['Life expectancy', 'Income composition of resources', 'Adult Mortality', 'Schooling']
    fig = sns.pairplot(df[selected_cols])
    st.pyplot(fig)

# Preprocessing
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
X = df_imputed.drop(columns=["Life expectancy"])
y = df_imputed["Life expectancy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_type = st.selectbox("Choose model", ["Random Forest", "Linear Regression"])

if model_type == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred)
    }

metrics = evaluate(y_test, y_pred)
st.subheader("📊 Model Evaluation")
st.json(metrics)

# Prediction Plot
st.subheader("📈 Actual vs Predicted Life Expectancy")
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolors='k')
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.grid(True)
st.pyplot(fig2)
