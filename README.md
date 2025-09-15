# Machine-Learning-Implementation-With-Framework

This repository contains an implementation of the **ID3 decision tree algorithm** using the **scikit-learn framework**. The project demonstrates how to build, train, and evaluate classification models with decision trees, one-hot encoding for categorical features, and additional improvements such as **GridSearchCV for hyperparameter tuning** and **Bagging** for better performance and generalization.  

The implementation includes:  
- Data preprocessing with one-hot encoding  
- Train/test split with automatic CSV export for reproducibility  
- Model training with entropy as the splitting criterion (ID3-style)  
- Cross-validation and hyperparameter tuning with GridSearchCV  
- Bias/variance diagnostics with explanatory text  
- Learning and validation curves  
- Confusion matrix (numeric and heatmap) and classification reports  
- Bagging for performance improvement and stability  

## Dataset Requirements
- Input datasets must be in CSV format.
- The target/label column must always be the final column in the dataset.
- All feature columns before the target will be treated as categorical variables and automatically one-hot encoded.

## Outputs
- Train/test CSV splits saved automatically (e.g., heart_train.csv, heart_test.csv).

### Console outputs including:

* Accuracy (train/test)
* Bias/variance diagnostics
* Confusion matrix and classification report

### Graphical outputs:

* Train vs. Test accuracy bar chart
* Learning curve
* Validation curve
* Confusion matrix heatmap

## Usage

Run the implementation from the command line:

```bash
python id3DecisionTree.py "dataset.csv" 