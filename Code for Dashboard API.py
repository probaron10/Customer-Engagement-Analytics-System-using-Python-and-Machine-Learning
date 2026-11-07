# app.py - Customer Engagement Analytics System using Python and Machine Learning

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_data(df):
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # One-hot encode categorical variables (excluding customerID, Churn)
    df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns.drop(['customerID','Churn']), drop_first=True)

    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    X = df.drop(['customerID','Churn'], axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Customer Engagement Analytics System")

uploaded_file = st.file_uploader("Upload Telco Customer Churn CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())

    # Preprocess
    X_scaled, y, df = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train models
    log_reg = LogisticRegression(max_iter=1000)
    dtree = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    log_reg.fit(X_train, y_train)
    dtree.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    st.success("✅ Models trained successfully!")

    # Compare models
    models = {"Logistic Regression": log_reg, "Decision Tree": dtree, "Random Forest": rf}
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred) * 100
        rec = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100
        results.append([name, acc, prec, rec, f1])

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"])
    st.write("### Model Comparison")
    st.dataframe(results_df)

    best_model = results_df.loc[results_df['Accuracy (%)'].idxmax()]
    st.write(f"✅ **Final Selected Model:** {best_model['Model']} (Accuracy: {best_model['Accuracy (%)']:.2f}%)")

    # Confusion Matrix for best model (Random Forest)
    cm = confusion_matrix(y_test, rf.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve (Random Forest)
    y_pred_prob = rf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC={auc:.2f}")
    ax2.plot([0,1],[0,1],'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve - Random Forest")
    ax2.legend()
    st.pyplot(fig2)

    # Feature Importance Plot
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = df.drop(['customerID','Churn'], axis=1).columns

    st.write("### Top 10 Important Features (Random Forest)")
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.barplot(x=features[indices[:10]], y=importances[indices[:10]], palette="coolwarm", ax=ax3)
    ax3.set_xticklabels(features[indices[:10]], rotation=45)
    ax3.set_ylabel("Importance")
    ax3.set_title("Top 10 Feature Importances")
    st.pyplot(fig3)

    # Correlation Heatmap
    st.write("### Feature Correlation Heatmap")
    numeric_df = df.drop(columns=['customerID'])
    corr = numeric_df.corr()
    fig4, ax4 = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax4)
    ax4.set_title("Feature Correlation Heatmap")
    st.pyplot(fig4)

    # Sample Prediction
    sample = X_test[0].reshape(1, -1)
    prediction = rf.predict(sample)[0]
    st.write("### Sample Prediction")
    st.write("Churn" if prediction == 1 else "No Churn")

    # Predict churn on the test set using Logistic Regression (or Random Forest)
    y_pred = log_reg.predict(X_test)   # change to rf.predict(X_test) if you prefer Random Forest
    y_pred_prob = log_reg.predict_proba(X_test)[:,1] * 100  # churn probability %

    # Get test customers
    test_customers = df.iloc[y_test.index].copy()
    test_customers.loc[:, 'PredictedChurn'] = y_pred
    test_customers.loc[:, 'ChurnProbability (%)'] = y_pred_prob

    # Show churned customers
    churned_customers = test_customers[test_customers['PredictedChurn'] == 1]

    st.write("### Customers Predicted to Churn")
    st.dataframe(churned_customers[['customerID', 'ChurnProbability (%)']].head(10))  # show first 10

    # Overall churn percentage
    churn_rate = (churned_customers.shape[0] / test_customers.shape[0]) * 100
    st.write(f"📊 Overall predicted churn rate: {churn_rate:.2f}% of customers in the test set")

    # Optional: Pie chart of churn vs non-churn
    fig5, ax5 = plt.subplots()
    labels = ['No Churn', 'Churn']
    sizes = [test_customers.shape[0] - churned_customers.shape[0], churned_customers.shape[0]]
    ax5.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightgreen','salmon'])
    ax5.set_title("Predicted Churn Distribution")
    st.pyplot(fig5)
