# German Credit Numeric Classification Project

This project focuses on building and evaluating machine learning models to classify creditworthiness using the German Credit Numeric dataset. The primary objective is to develop a robust classification system that distinguishes between good and bad credit risks.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Modeling Approach](#modeling-approach)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [Acknowledgements](#acknowledgements)

## Project Overview

The German Credit Numeric dataset, sourced from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/german_credit_numeric), contains anonymized data on credit applications and associated outcomes. The project employs Logistic Regression and Dense Neural Network (DNN) classifiers to predict creditworthiness. Additionally, hyperparameter tuning is conducted using TensorBoard's HParams Dashboard to enhance model performance.

## Dataset

The German Credit Numeric dataset includes 1,000 instances with 24 numerical features and a binary target variable indicating whether the credit applicant is considered a good (1) or bad (0) credit risk.

### Features

- **Numerical features**: 24 anonymized attributes related to the applicant’s financial status and credit history.
- **Target variable**: Binary classification indicating creditworthiness (`1` for good, `0` for bad).


## Modeling Approach

### 1. Logistic Regression

A Linear Classifier using Scikit-Learn's Logistic Regression, which performs linear regression with a softmax output.

- **Cross-Validation**: Applied k-fold cross-validation to assess the model’s stability and performance across different subsets of the data.
- **Evaluation**: Evaluated using ROC AUC to measure the classifier's ability to distinguish between good and bad credit risks.

### 2. Dense Neural Network (DNN)

A Dense Neural Network classifier was implemented using TensorFlow and evaluated with TensorBoard for hyperparameter tuning.

- **Hyperparameter Tuning**: Utilized TensorBoard's HParams Dashboard to explore various settings, including dropout rate, number of layers, units per layer, and optimizer types.
- **Best Hyperparameters**: The highest AUC value of 0.94637 on the validation set was achieved with:
  - Dropout = 0.0
  - Number of layers = 3
  - Units per layer = 48
  - Optimizer = 'adam'
  
  However, this configuration did not generalize well to the test set.

### 3. Abstract Base Class for Learning Algorithms

- **BaseLearningAlgorithm**: An abstract base class defining `train()`, `predict()` methods, and a `name` property to label the plots.
- **LogisticRegressionLearningAlgorithm**: A simple wrapper class around Scikit-Learn’s Logistic Regression implementing the base class.

## Evaluation

The project focuses on evaluating model performance using ROC AUC, a key metric for binary classifiers, particularly in imbalanced datasets. The evaluation process includes:

- **ROC AUC Enhancement**: Enhanced ROC AUC plots to display results from multiple cross-validation folds for better visualization and comparison.
- **Comparison**: Compared the performance of Logistic Regression against DNN classifiers to identify the most effective model.

## Results

### Logistic Regression

- **Best ROC-AUC**: 0.756

### Dense Neural Network

- **Highest AUC on Validation**: 0.94637 (dropout = 0.0, num_layers = 3, num_units = 48, optimizer = 'adam')
- **Best Test AUC**: 0.790 with settings:
  - Dropout = 0.15
  - Number of layers = 2
  - Units per layer = 48
  - Optimizer = 'rmsprop'

Despite extensive tuning, the best performance on the test set was still lower than that achieved with Logistic Regression, suggesting potential overfitting or the need for further exploration of hyperparameter combinations.

## Acknowledgements

- This project uses the German Credit Numeric dataset from TensorFlow Datasets.
