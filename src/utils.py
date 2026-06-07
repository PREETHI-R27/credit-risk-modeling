"""Utility functions for Credit Risk Modeling."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper


def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method."""
    z = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z > threshold]


def plot_feature_distributions(df, numerical_cols, save_path='../reports/feature_distributions.png'):
    """Plot distributions for numerical features."""
    n = len(numerical_cols)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4*rows))
    axes = axes.ravel()

    for idx, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color='steelblue')
        axes[idx].set_title(f'Distribution of {col}')

    # Hide extra subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_summary_report(df, target_col='Credit_risk'):
    """Generate a text summary report of the dataset."""
    report = []
    report.append(f"Dataset Shape: {df.shape}")
    report.append(f"Target Distribution:")
    report.append(str(df[target_col].value_counts(normalize=True)))
    report.append(f"\nMissing Values:")
    report.append(str(df.isnull().sum().sum()))
    report.append(f"\nDuplicated Rows: {df.duplicated().sum()}")
    return "\n".join(report)


if __name__ == "__main__":
    print("Testing Utility Functions...")
    df_test = pd.DataFrame({
        'Age': [25, 30, 35, 100],  # 100 is an outlier
        'Credit_risk': [0, 1, 0, 1]
    })
    
    outliers, low, up = detect_outliers_iqr(df_test, 'Age')
    print(f"Detected {len(outliers)} outliers in Age")
    
    report = generate_summary_report(df_test)
    print("\nSummary Report Preview:")
    print(report)
