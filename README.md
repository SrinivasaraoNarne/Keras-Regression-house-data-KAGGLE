# Keras Regression - House Price Prediction
## Introduction
This project demonstrates how to build and train a Keras-based Artificial Neural Network (ANN) to predict house prices using a dataset from Kaggle. The dataset includes features such as square footage, number of bedrooms, bathrooms, location, and other property attributes to enhance predictive accuracy.
## Overview
The project aims to develop a robust regression model using Keras, leveraging historical housing data to generate reliable price predictions. Key aspects of the project include:
* Data preprocessing and feature engineering.
* Constructing and training a regression ANN model.
* Evaluating model performance using key metrics like Mean Squared Error (MSE) and R².
* Deploying the model for future predictions.
## Dataset
The dataset used in this project is sourced from Kaggle and contains the following features:
  * Numerical features: sqft_living, sqft_lot, bedrooms, bathrooms, etc.
  * Categorical features: zipcode, view, condition, etc.
  * Target variable: price (house price in USD).
  * Data: [Kaggle](https://www.kaggle.com/)
## Project Workflow
1. Data Preprocessing:
   * Handled missing values.
   * Applied feature scaling and normalization.
   * Performed exploratory data analysis (EDA).
2. Model Building:
   * Created an ANN with Keras for regression.
   * Optimized hyperparameters such as learning rate, batch size, and number of layers.
3. Evaluation:
   * Evaluated the model using metrics like MSE, RMSE, and R².
4. Prediction:
   * Used the trained model to predict house prices on test data.
## Technologies Used
  * Programming Language: Python
  * Libraries: Keras, TensorFlow, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
  * Environment: Jupyter Notebook / Python IDE
  * The above written code was using Jupyter Notebook

