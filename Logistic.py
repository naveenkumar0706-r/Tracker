# log_reg_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.title("Titanic Survival Prediction - Logistic Regression")

# -------------------------------------------------
# 1) Load dataset
# -------------------------------------------------
df = pd.read_csv(r"C:\Users\navee\Desktop\Data_Scientist\Project\titanic.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# 2) Preprocessing
# -------------------------------------------------
df = df[[
    "survived", "p_class", "sex", "age",
    "sib_sp", "parch", "fare", "embarked"
]].dropna()

# Encode categorical columns
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["embarked"] = df["embarked"].map({"C": 0, "Q": 1, "S": 2})

X = df[[
    "p_class", "sex", "age",
    "sib_sp", "parch", "fare", "embarked"
]]
y = df["survived"]

# -------------------------------------------------
# 3) Train model
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.write(f"Train Accuracy: {model.score(X_train, y_train):.3f}")
st.write(f"Test Accuracy: {model.score(X_test, y_test):.3f}")

# -------------------------------------------------
# 4) User Input
# -------------------------------------------------
st.subheader("Try Your Own Passenger")

p_class = st.selectbox("Passenger Class", [1, 2, 3], index=2)
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sib_sp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

sex_val = 1 if sex == "male" else 0
emb_val = {"C": 0, "Q": 1, "S": 2}[embarked]

# -------------------------------------------------
# 5) Prediction
# -------------------------------------------------
if st.button("Predict Survival"):
    X_new = [[
        p_class, sex_val, age,
        sib_sp, parch, fare, emb_val
    ]]

    pred_prob = model.predict_proba(X_new)[0][1]
    pred = model.predict(X_new)[0]

    st.write(f"Survival Probability: {pred_prob:.3f}")

    if pred == 1:
        st.success("Passenger Survived ✅")
    else:
        st.error("Passenger Did Not Survive ❌")
