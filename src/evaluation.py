import numpy as np

def regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics: RMSE, MAE, and R2 score.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns:
    dict: A dictionary containing RMSE, MAE, and R2 score.
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }