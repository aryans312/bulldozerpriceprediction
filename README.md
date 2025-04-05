# Bulldozer Price Prediction
Description
This project focuses on predicting the resale prices of used bulldozers using machine learning techniques. The data comes from auction listings and includes various features such as model ID, manufacturing year, equipment type, and usage hours. The aim is to build a regression model that can accurately estimate the selling price of a bulldozer based on its attributes.

It demonstrates a full machine learning pipeline, including:

Data loading and preprocessing

Time series handling

Feature engineering and selection

Model training and evaluation using Random Forest

Saving the trained model for future use

The project is based on the well-known Kaggle Bluebook for Bulldozers competition.

Project Structure
end-to-end-bluebook-bulldozer-price-regression-v2.ipynb: Main Jupyter notebook with complete ML pipeline.

data/: Directory for input datasets (optional, if you're storing locally).

models/: Folder to save trained models (.pkl files).

README.md: Project overview and setup instructions.

Tech Stack
Python: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

Jupyter Notebook

Machine Learning: Random Forest Regressor

RMSLE (Root Mean Squared Log Error) for model evaluation

Problem Statement
Predict the sale price of a bulldozer given its features such as model, year of manufacture, equipment type, and usage details. This helps in accurate asset valuation for resale or purchase decisions.

Key Concepts
Parsing and working with date-based features (saledate)

Handling missing values and categorical data

Evaluating models using RMSLE

Understanding feature importance

Exporting the trained model using joblib

How to Run
Clone this repository-
git clone https://github.com/aryans312/bulldozerpriceprediction

Install dependencies-
pip install -r requirements.txt
Launch the notebook
jupyter notebook
(Optional) Use the trained model in production


import joblib
model = joblib.load("models/bulldozer-model.pkl")

Project Status-
Completed core ML pipeline with good performance
Potential enhancements: Hyperparameter tuning, Streamlit dashboard, deployment on cloud platform

Credits
This project is based on the Kaggle Bluebook for Bulldozers competition.

Special thanks to Andrei Neagoie and Daniel Bourke for their excellent instruction and guidance through their Udemy course. This project was completed as part of their machine learning curriculum.

