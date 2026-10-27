# AERONEX — Weather-Based Flight Safety Prediction

## Overview

AERONEX is a machine learning mini-project that analyzes weather-related features and classifies flight weather conditions into **Safe** and **Unsafe** categories. It provides an interactive dashboard built with Streamlit for entering weather parameters, viewing predictions, and exploring the dataset.

## Features

- Interactive and modern Streamlit dashboard
- Weather-based Safe/Unsafe classification
- Weather parameter input and prediction
- Interactive data visualizations
- Dataset exploration
- Trained machine learning model integration
- Model development using cross-validation and hyperparameter tuning

## Technologies Used

- **Programming Language:** Python
- **Frontend and Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualizations:** Plotly
- **Model Serialization:** Joblib

## Project Structure

```text
Weather_Flight_Project/
├── app.py
├── flight_weather_safety_model.joblib
├── weather_flight_safety.csv
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd weather-flight-project
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your GitHub repository URL.

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to access the AERONEX dashboard.

## Dataset and Model

The project uses the `weather_flight_safety.csv` dataset and a trained model saved as `flight_weather_safety_model.joblib`.

The application processes the selected weather parameters and uses the trained machine learning model to generate a classification result.

## Model Development

The project workflow includes data preparation, model training, cross-validation, and hyperparameter tuning to evaluate and improve model performance.

## Limitations

This is an educational machine learning prototype. The dataset uses synthetic, rule-generated labels, and the predictions have not been validated as assessments of real-world aviation safety. The application must not be used to make actual flight or operational safety decisions.

## Future Enhancements

- Incorporate verified real-world weather and aviation datasets
- Evaluate performance on independently validated data
- Add model explainability and feature importance analysis
- Expand dashboard analytics and reporting capabilities

## Project Objective

To demonstrate how machine learning, data analysis, cross-validation, and hyperparameter tuning can be applied to a weather-based classification problem through an interactive web application.

---

**Project Name:** AERONEX  
**Project Type:** University Mini-Project  
**Domain:** Machine Learning and Data Science
