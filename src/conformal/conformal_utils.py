import numpy as np
from collections import deque
import json
from pathlib import Path

class ConformalRegressor:
    """
    Lightweight Conformal Prediction module for streaming inference.
    Maintains a rolling calibration window of residuals.
    """

    def __init__(self, alpha=0.1, window_size=500):
        self.alpha = alpha
        self.window_size = window_size
        self.residuals = deque(maxlen=window_size)
        self.calibrated = False

    def fit_calibration(self, y_true, y_pred):
        """Compute nonconformity scores on calibration data."""
        residuals = np.abs(y_true - y_pred)
        self.residuals.extend(residuals.tolist())
        self.calibrated = True
        return self

    def update(self, y_true, y_pred):
        """Update residuals dynamically (streaming mode)."""
        residuals = np.abs(y_true - y_pred)
        self.residuals.extend(residuals.tolist())

    def get_quantile(self):
        """Return current (1 - alpha) quantile for uncertainty."""
        if not self.residuals:
            return 0.1  # fallback
        return np.quantile(list(self.residuals), 1 - self.alpha)

    def predict_with_uncertainty(self, model, X_new):
        """Predict mean + conformal uncertainty (half-width)."""
        y_pred = model.predict(X_new)
        q = self.get_quantile()
        lower = y_pred - q
        upper = y_pred + q
        y_std = np.full_like(y_pred, q)
        return y_pred, y_std, lower, upper

    def save(self, path):
        """Save current calibration state."""
        data = {"alpha": self.alpha, "residuals": list(self.residuals)}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        """Load saved calibration."""
        obj = cls()
        data = json.loads(Path(path).read_text())
        obj.alpha = data["alpha"]
        obj.residuals = deque(data["residuals"], maxlen=obj.window_size)
        obj.calibrated = True
        return obj
