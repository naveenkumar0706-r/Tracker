import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="KNN Social Network Ads", layout="wide")
st.title("📊 Social Network Ads - KNN Classifier")

uploaded = st.file_uploader("Upload Social_Network_Ads.csv", type=["csv"])

if uploaded:
    data = pd.read_csv(uploaded)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    X = data[["Age", "EstimatedSalary"]]
    y = data["Purchased"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    st.sidebar.header("Model Settings")
    k = st.sidebar.slider("Number of Neighbors (k)", 1, 30, 5)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    col1, col2 = st.columns(2)
    col1.metric("Train Accuracy", f"{model.score(X_train_sc, y_train):.2f}")
    col2.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("🔍 Predict New Customer")
    age = st.number_input("Age", 18, 70, 35)
    salary = st.number_input("Estimated Salary", 15000, 200000, 50000)

    if st.button("Predict"):
        sample = scaler.transform([[age, salary]])
        result = model.predict(sample)[0]
        st.success("Purchased" if result == 1 else "Not Purchased")

    st.subheader("📈 Decision Boundary")
    x_min, x_max = X["Age"].min() - 2, X["Age"].max() + 2
    y_min, y_max = X["EstimatedSalary"].min() - 5000, X["EstimatedSalary"].max() + 5000

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.5),
        np.arange(y_min, y_max, 3000),
    )

    grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[y == 0]["Age"], X[y == 0]["EstimatedSalary"], c="red", s=10, label="Not Purchased")
    ax.scatter(X[y == 1]["Age"], X[y == 1]["EstimatedSalary"], c="green", s=10, label="Purchased")
    ax.set_xlabel("Age")
    ax.set_ylabel("Salary")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Please upload Social_Network_Ads.csv")
