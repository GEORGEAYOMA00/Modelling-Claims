# Modelling Car Insurance Claim Outcomes
# Import required modules
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit

# Read in dataset
cars = pd.read_csv("car_insurance.csv")

# Check for missing values
cars.info()

# Fill missing values with the mean
cars["credit_score"].fillna(cars["credit_score"].mean(), inplace=True)
cars["annual_mileage"].fillna(cars["annual_mileage"].mean(), inplace=True)

# Empty list to store model results
models = []

# Feature columns
features = cars.drop(columns=["id", "outcome"]).columns

# Loop through features
for col in features:
    # Create a model
    model = logit(f"outcome ~ {col}", data=cars).fit()
    # Add each model to the models list
    models.append(model)

# Empty list to store accuracies
accuracies = []

# Loop through models
for feature in range(0, len(models)):
    # Compute the confusion matrix
    conf_matrix = models[feature].pred_table()
    # True negatives
    tn = conf_matrix[0,0]
    # True positives
    tp = conf_matrix[1,1]
    # False negatives
    fn = conf_matrix[1,0]
    # False positives
    fp = conf_matrix[0,1]
    # Compute accuracy
    acc = (tn + tp) / (tn + fn + fp + tp)
    accuracies.append(acc)

# Find the feature with the largest accuracy
best_feature = features[accuracies.index(max(accuracies))]

# Create best_feature_df
best_feature_df = pd.DataFrame({"best_feature": best_feature,
                                "best_accuracy": max(accuracies)},
                                index=[0])
best_feature_df

An insurance company operating in a large and competitive market where car insurance is legally required, sought to optimize their pricing and risk assessment by predicting whether a customer would make a claim during the policy period. Due to limited expertise and infrastructure for deploying machine learning models, they requested a simple, single-feature model that could deliver the highest accuracy. My role was to analyze their customer data, identify the most predictive feature, and provide actionable insights to guide their initial model deployment.

Key Insights: 
The analysis revealed that driving experience is the single most predictive feature, achieving an accuracy of 77.71%.
Impact and Implications: The project’s impact lies in its ability to enable a simplified and cost-effective model deployment, focusing on a single, interpretable feature that requires minimal infrastructure and technical expertise. By identifying driving experience as the most predictive feature, the company can refine its pricing strategies to offer competitive rates to low-risk customers while mitigating exposure to high-risk groups. Additionally, the findings provide a foundation for customer segmentation, allowing for targeted marketing and personalized policy offerings based on driving experience. While the model was designed for simplicity, it also establishes a scalable framework for future enhancements, enabling the company to incorporate additional features as their infrastructure and expertise grow, further improving predictive performance.

Approach:
Data Exploration: Loaded and explored the dataset (car_insurance.csv) to understand its structure, identify key variables, and assess data quality.
Data Cleaning: Addressed missing values to ensure the dataset was complete and ready for modeling. Missing data was handled using appropriate imputation techniques to maintain data integrity and avoid bias.
Model Preparation: Built univariate logistic regression models for each feature, predicting the likelihood of a claim (outcome) based on that feature alone. Stored all models in a models list for further evaluation.
Performance Measurement: Evaluated the performance of each model by computing the confusion matrix and calculating the accuracy for each feature. Accuracy was chosen as the primary metric to align with the company’s goal of deploying a simple and reliable model.
Feature Selection: Identified the feature with the highest accuracy as the best-performing predictor. Created a summary DataFrame to clearly communicate the results, highlighting the top-performing feature and its corresponding accuracy.

Conclusion: This project demonstrates my ability to approach business challenges with a highly analytical mindset, leveraging data to deliver actionable insights. By identifying driving experience as the most predictive feature, I provided the company with a practical, high-impact solution to improve their pricing and risk assessment processes. This work not only addressed their immediate needs but also laid the groundwork for future advancements in their data-driven decision-making capabilities.
