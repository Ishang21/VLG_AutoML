# VLG_AutoML

# Automated Hyperparameter Optimization

## Project Overview

The quality of performance of a Machine Learning model heavily depends on its hyperparameter settings. This project aims to develop an automated hyperparameter optimization (HPO) system using AutoML techniques that can efficiently identify the best hyperparameter configuration for a given machine learning model and dataset. The system is designed to integrate with various machine learning models and handle different data types.

### Key Features
- **Automated Hyperparameter Optimization**: Implemented Bayesian optimization for hyperparamter Optimization and compared the results with Randomized Search and Hyperopt scores.
- **Model Evaluation**: Comparison of ROC AUC, cross-validation, and learning rate distribution curves across different optimization techniques.
- **Versatile Integration**: Capable of integrating with various machine learning models.

## Repository Structure

├── README.md
├── requirements.txt
├── bayesian_optimizer.py
├── Randomized_Search.py
├── Hyperopt_optimizer.py
├── models.py
├── main.py
├── evaluation.py
├── data/
│ └── dataset.csv


- `README.md`: This file.
- `requirements.txt`: List of required Python packages.
- `bayesian_optimizer.py`: Implementation of Bayesian optimization for hyperparameter tuning.
- `Randomized_Search.py`: Implementation of Randomized Search for hyperparameter tuning.
- `Hyperopt_optimizer.py`: Custom Hyperopt-like optimization implementation.
- `models.py`: Contains functions to define and retrieve machine learning models.
- `main.py`: Main script to run the entire pipeline, from data preprocessing to model evaluation.
- `evaluation.py`: Contains functions for evaluating model performance and comparing different models.
- `data/dataset.csv`: Sample dataset used for model training and evaluation.

## Installation

To set up the project, clone the repository and install the required packages:

```sh
git clone https://github.com/Ishang21/VLG_AutoML.git
cd VLG_AutoML/src
pip install -r requirements.txt


Usage
Prepare the Dataset: Ensure that your dataset is placed in the data/ directory and is named dataset.csv.

Run the Main Script: Execute the main.py script to start the hyperparameter optimization and model evaluation process.

Hyperparameter Optimization Techniques
Bayesian Optimization
Bayesian optimization is an efficient method for finding the optimum of a function that is expensive to evaluate. It uses a probabilistic model to make predictions about the function's behavior and selects the most promising points to evaluate next.

Randomized Search
Randomized Search involves sampling hyperparameter combinations at random from a specified range. It is a simple and effective method for hyperparameter tuning, especially when the parameter space is large.

Hyperopt 
It is a built-in Library for automated hyperparamter optimization.

