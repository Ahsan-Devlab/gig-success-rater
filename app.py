import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Gig Success Rater", page_icon="💼")
st.title("💼 Freelance Gig Success Rater")
st.write("Understand Random Forest and Feature Importance.")



@st.cache_data
def load_and_train():
    df = pd.read_csv('freelance_data.csv')
    
    X = df[['budget', 'deadline_days', 'client_history']]
    y = df['status']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    return model, df

model, df = load_and_train()


st.sidebar.header("Enter Project Details")

user_budget = st.sidebar.number_input("Project Budget ($)", min_value=10, max_value=10000, value=500)
user_deadline = st.sidebar.slider("Deadline (Days)", min_value=1, max_value=60, value=14)
user_history = st.sidebar.slider("Client History Rating", min_value=1, max_value=5, value=3, help="1 = Bad history, 5 = Great history")


prediction = model.predict([[user_budget, user_deadline, user_history]])[0]

st.subheader("Prediction Result")
if prediction == 1:
    st.success("🟢 **On Time:** This freelance project is likely to be completed successfully!")
else:
    st.error("🔴 **Late / At Risk:** This project has a high risk of failure, delay, or cancellation.")

st.subheader("Why did the AI make this choice?")

importances = model.feature_importances_
features = ['Budget', 'Deadline Days', 'Client History']

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(features, importances, color=['#3498db', '#e74c3c', '#2ecc71'])
ax.set_ylabel("Importance Score (0 to 1)")
ax.set_title("Feature Importance in Random Forest")
ax.grid(axis='y', linestyle='--', alpha=0.7)

st.pyplot(fig)