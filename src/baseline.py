from sklearn.dummy import DummyRegressor
from evaluation import regression_metrics

def train_and_evaluate(X_train, y_train, X_val, y_val):
    # Initialize a DummyRegressor with the "mean" strategy.
    model = DummyRegressor(strategy="mean")

    # Fit the model on the training data.
    model.fit(X_train, y_train)

    # Predict on the validation set.
    y_pred = model.predict(X_val)

    # Calculate regression metrics.
    metrics = regression_metrics(y_val, y_pred)

    return model, metrics