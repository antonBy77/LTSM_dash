# predictor.py

import numpy as np
from tensorflow.keras.models import load_model # type: ignore

def make_future_predictions(model, last_sequence, future_steps, scaler):
    """
    Прогнозирует будущие значения с использованием модели LSTM.
    """
    future_predictions = []

    for _ in range(future_steps):
        X_last = last_sequence.reshape((1, len(last_sequence), 1))
        future_pred = model.predict(X_last)
        future_predictions.append(future_pred[0, 0])

        last_sequence = np.append(last_sequence[1:], future_pred)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions