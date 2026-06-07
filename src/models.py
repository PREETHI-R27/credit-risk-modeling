"""Model training and evaluation utilities for Credit Risk Modeling."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, accuracy_score, f1_score
)
import joblib


def train_logistic_regression(X_train, y_train, class_weight='balanced'):
    """Train logistic regression baseline."""
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=10, class_weight='balanced'):
    """Train Random Forest ensemble."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        random_state=42,
        class_weight=class_weight,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=200, max_depth=4, learning_rate=0.1):
    """Train XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """Evaluate model and return metrics dictionary."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob),
        'Avg_Precision': average_precision_score(y_test, y_prob),
        'False_Negative_Rate': fn_rate,
        'Confusion_Matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    return metrics


def plot_roc_curves(results_dict, save_path='../reports/roc_auc_comparison.png'):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'purple']

    for idx, (name, metrics) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_prob'])
        auc = metrics['ROC_AUC']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', 
                color=colors[idx % len(colors)], linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(results_dict, save_path='../reports/confusion_matrices.png'):
    """Plot confusion matrices side by side."""
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    for idx, (name, metrics) in enumerate(results_dict.items()):
        cm = metrics['Confusion_Matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Predicted Good', 'Predicted Bad'],
                   yticklabels=['Actual Good', 'Actual Bad'])
        axes[idx].set_title(name)

    plt.suptitle('Confusion Matrices - False Negatives are Costly!', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_model(model, filepath='../models/xgboost_credit_risk.pkl'):
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath='../models/xgboost_credit_risk.pkl'):
    """Load trained model from disk."""
    return joblib.load(filepath)


if __name__ == "__main__":
    print("Testing Model Training Utilities...")
    from data_prep import preprocess_pipeline
    
    try:
        X_train, X_test, y_train, y_test, scaler = preprocess_pipeline('../data/german_credit.csv')
        print("Data loaded. Training quick XGBoost model...")
        model = train_xgboost(X_train, y_train, n_estimators=10)
        metrics = evaluate_model(model, X_test, y_test, "Quick XGB")
        print(f"Model Accuracy: {metrics['Accuracy']:.4f}")
        print(f"ROC AUC: {metrics['ROC_AUC']:.4f}")
    except Exception as e:
        print(f"Error during testing: {e}")
