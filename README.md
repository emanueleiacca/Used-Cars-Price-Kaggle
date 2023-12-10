# Craigslist Used Cars Dataset Regression Ensemble

## Overview
This repository encompasses a detailed analysis of various regression models and their ensemble for predicting prices based on a Craigslist Used Cars dataset. The dataset consists of approximately 200,000 rows with 26 features, including Price, Year, Manufacturer, Model, Odometer, and more.

## Data Preparation Steps
- **Null Check:** No null values found.
- **Duplicate Removal:** Eliminated duplicates based on key columns.
- **Type Conversion:** Converted relevant columns to numeric types.
- **Outlier Handling:** Removed outliers in 'price' and 'odometer.'
- **Feature Scaling:** Explored scaling techniques (Robust Scaler).
- **Log Transformation:** Applied log transformation for skewed variables.
- **Correlation Analysis:** Explored correlations for post-/pre-1976 cars.
- **Column Removal:** Dropped irrelevant columns.
- **Label Encoding:** Converted categorical to numerical using Label Encoding.
- **Missing Values:** Removed rows with missing values.

## Regression Model Ensemble

This project aims to predict prices using various regression models, including linear regression, random forest regression, polynomial regression, support vector machines, and stochastic gradient descent. Additionally, Optuna, a hyperparameter optimization framework, is employed to fine-tune the models for enhanced performance.

## Table of Contents

- [Introduction](#introduction)
- [Models Examined](#models-examined)
- [Optimization with Optuna](#Optimization-with-Optuna)
- [Models with Optuna](#Models-with-Optuna)
- [Results](#results)

## Introduction

In this project, we explore different regression models, including linear regression, random forest regression, polynomial regression, support vector machines, and stochastic gradient descent, to predict prices. Optuna is used for hyperparameter tuning to optimize model performance.

## Models Examined

1. **Linear Regression**
   - Fast and interpretable.
   - Assumes a linear relationship between independent variables and the target variable.

2. **Random Forest Regressor**
   - Suitable for predicting continuous numeric values.
   - Ensemble of decision trees.

3. **Polynomial Regression**
   - Introduces polynomial terms to capture nonlinear relationships.

4. **Support Vector Machines (SVM)**
   - Effective in capturing nonlinear relationships.
   - Utilizes different kernels (e.g., RBF, polynomial, linear).

5. **Stochastic Gradient Descent**
   - Linear model trained using stochastic gradient descent.

## Optimization with Optuna

This project employs Optuna to fine-tune hyperparameters for each model, including Random Forest, XGBoost, LightGBM, and AdaBoost. The ensemble model, combining Random Forest, XGBoost, and LightGBM, is further optimized by adjusting the weights assigned to each model.

## Models with Optuna

1. **Random Forest Regression**
   - The Random Forest model's hyperparameters, such as the number of estimators, maximum depth, and minimum samples split, are optimized using Optuna to achieve the best possible performance.

2. **XGBoost Regression**
   - Optuna is utilized to fine-tune hyperparameters like the booster type, maximum depth, learning rate, and the number of estimators for the XGBoost regression model.

3. **LightGBM Regression**
   - Similar to XGBoost, the LightGBM model's hyperparameters, including the number of leaves, learning rate, feature fraction, and bagging fraction, are optimized using Optuna.

4. **AdaBoost Regression**
   - Similar to XGBoost, the LightGBM model's hyperparameters, including the number of leaves, learning rate, feature fraction, and bagging fraction, are optimized using Optuna.

5. **Ensemble Weight Optimization**
   - The ensemble model, combining Random Forest, XGBoost, and LightGBM, is further optimized by adjusting the weights assigned to each model. Optuna is employed to find the optimal ensemble weight configuration.



## Results

The README presents the best hyperparameters obtained for each model and provides an overall R-squared score for the final ensemble. Additionally, performance metrics such as Mean Squared Error, R-squared, Mean Absolute Error, and Root Mean Squared Error are visualized for easy comparison between models.

![image](https://github.com/emanueleiacca/Used-Cars-Price-Kaggle/assets/128679981/7c9527fd-0216-460a-8a59-8245ce739a0f)


