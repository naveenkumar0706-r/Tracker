import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("🎓 Student GPA Regression Demo")


uploaded_file = st.file_uploader(
    "Browse and upload student_details.csv", 
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df)

    # 2. Train–test split
    X = df.drop("GPA", axis=1)
    y = df["GPA"]

    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.4, random_state=50
    )

    
    scaler = StandardScaler()
    X_train_ss = scaler.fit_transform(X_train)
    X_test_ss = scaler.transform(X_test)

    
    model = LinearRegression()
    model.fit(X_train_ss, y_train)

    
    y_pred = model.predict(X_test_ss)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"Mean Squared Error (MSE): **{mse:.4f}**")
    st.write(f"R² Score: **{r2:.4f}**")

   
    st.subheader("Correlation (Age, Credits, GPA)")
    st.write(df[["Age", "Credits_Completed", "GPA"]].corr())

    
    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=np.abs, ascending=False)

    st.subheader("Top 10 Features by Coefficient")
    st.dataframe(coef_df.head(10))

    
    st.subheader("Actual vs Predicted GPA")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()]
    )
    ax.set_xlabel("Actual GPA")
    ax.set_ylabel("Predicted GPA")
    ax.set_title("Actual vs Predicted GPA")
    ax.grid(True)

    st.pyplot(fig)

else:
    st.info("👆 Please browse and upload the CSV file to continue.")
