# Credit Risk Modeling

## Dataset Information
- **Source**: German Credit Dataset
- **Samples**: 1,000 credit applicants
- **Target**: Credit_risk (0 = Good, 1 = Bad)
- **Default Rate**: 30%

## Model Information
- **Algorithm**: XGBoost Classifier
- **Architecture**: Gradient Boosting with 200 estimators, max depth 4, learning rate 0.1.
- **Preprocessing**: Feature engineering, one-hot encoding, standard scaling, and SMOTE oversampling to handle class imbalance.

## Results
| Metric | Value |
|--------|-------|
| **ROC AUC** | **0.81** |
| **Accuracy** | **0.77** |
| **F1 Score** | **0.65** |
| **False Negative Rate** | **0.24** |

## Project Structure (Upload Focus)
- `src/`: Contains core logic for data preparation, modeling, and utility functions.
