# Early Diabetes Prediction

An end-to-end Machine Learning web application that predicts the likelihood of diabetes based on clinical health parameters. 
This repository contains an end-to-end machine learning project to predict the likelihood of diabetes using user-provided health data. 

## Project Overview

The goal of this project is to create a seamless process for predicting diabetes by building a machine learning model that analyzes various health parameters. The web application takes user input, processes the data through the model, and provides the prediction result.

## Project Objectives

<img width="512" height="650" alt="image" src="https://github.com/user-attachments/assets/793ada3f-732a-458c-89bd-8940a704a143" />

## Dataset

- Diabetes Dataset from Kaggle
- 8 clinical features
- Binary classification problem (Diabetic / Not Diabetic)

## ML Workflow
- **Tech stack**: Python, scikit-learn, pandas, numpy, matplotlib, seaborn, streamlit, github
- **Algorithms Used**: Logistic Regression, Decision Trees, Random Forests, SVM (or any chosen algorithms based on your requirement)
- **Input Features**: The following fields are taken from the user:
  - Number of Pregnancies
  - Insulin Level
  - Age
  - Body Mass Index (BMI)
  - Blood Pressure
  - Glucose Level
  - Skin Thickness
  - Diabetes Pedigree Function
- **Output**: The model predicts whether the person is likely to have diabetes (Yes/No).

**Note:** Some research-specific features (e.g., Diabetes Pedigree Function) were redesigned into user-friendly inputs for real-world usability.

## Deployment

The application is deployed using Streamlit Community Cloud.
Users can:

  * Input health parameters
  * Receive diabetes risk prediction
  * View probability score
  * Understand feature explanations

🔗 Live App: [https://gayathriramu-diabetes-prediction-app.streamlit.app/ ](https://early-diabetes-prediction-app-gayathriramu.streamlit.app/)

**⚠️ Disclaimer**
This tool is for educational purposes only and does not provide medical diagnosis.
