import numpy as np
from scipy.stats import spearmanr
import json

def conservative_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    # "higher" => conservative, ensures >= nominal coverage
    try:
        return float(np.quantile(abs_residuals, 1 - alpha, method="higher"))
    except TypeError:
        return float(np.percentile(abs_residuals, 100*(1-alpha), interpolation="higher"))

class SplitConformalRegressor:
    """
    Enhanced split-conformal for regression with comprehensive uncertainty analysis:
      - fit any model on train
      - get residuals on a held-out calibration set
      - compute q_{1-alpha}
      - prediction interval at test: [yhat - q, yhat + q]
      - includes epistemic uncertainty (MC Dropout) and aleatoric uncertainty (MSE-based)
    """
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_ = None
        self.epistemic_std = None
        self.aleatoric_stats = None

    def calibrate(self, y_calib_hat: np.ndarray, y_calib_true: np.ndarray):
        res = np.abs(y_calib_true - y_calib_hat)
        self.q_ = conservative_quantile(res, self.alpha)

    def calibrate_with_uncertainty(self, y_calib_hat: np.ndarray, y_calib_true: np.ndarray, 
                                 epistemic_std: np.ndarray = None):
        """Calibrate conformal prediction with additional uncertainty measures"""
        res = np.abs(y_calib_true - y_calib_hat)
        self.q_ = conservative_quantile(res, self.alpha)
        
        # Store epistemic uncertainty if provided
        if epistemic_std is not None:
            self.epistemic_std = epistemic_std
        
        # Compute aleatoric uncertainty (MSE-based)
        residuals = y_calib_true - y_calib_hat
        squared_residuals = residuals ** 2
        
        self.aleatoric_stats = {
            'mse': float(np.mean(squared_residuals)),
            'rmse': float(np.sqrt(np.mean(squared_residuals))),
            'mae': float(np.mean(np.abs(residuals))),
            'aleatoric_variance': float(np.var(residuals)),
            'residual_std': float(np.std(residuals))
        }

    def interval(self, y_hat: np.ndarray):
        if self.q_ is None:
            raise RuntimeError("Conformal not calibrated. Call calibrate() first.")
        L = y_hat - self.q_
        U = y_hat + self.q_
        return L, U

    def compute_coverage(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Compute empirical coverage of conformal intervals"""
        if self.q_ is None:
            raise RuntimeError("Conformal not calibrated.")
        
        L, U = self.interval(y_pred)
        covered = (y_true >= L) & (y_true <= U)
        coverage_rate = float(np.mean(covered))
        interval_width = float(np.mean(U - L))
        
        return {
            'overall_coverage': coverage_rate,
            'nominal_coverage': 1 - self.alpha,
            'interval_width': interval_width,
            'n_samples': len(y_true),
            'n_covered': int(np.sum(covered))
        }

    def analyze_uncertainty_correlation(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Analyze correlation between uncertainty measures and prediction errors"""
        residuals = np.abs(y_true - y_pred)
        correlations = {}
        
        # Correlation between epistemic uncertainty and prediction error
        if self.epistemic_std is not None:
            rho_epistemic, p_epistemic = spearmanr(residuals, self.epistemic_std)
            correlations['epistemic_vs_error'] = {
                'spearman_rho': float(rho_epistemic),
                'p_value': float(p_epistemic)
            }
        
        return correlations

    def get_comprehensive_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Generate comprehensive uncertainty report"""
        report = {
            'uncertainty_types': {
                'epistemic': {
                    'description': 'Model parameter uncertainty (MC Dropout)',
                    'available': self.epistemic_std is not None,
                    'mean_std': float(np.mean(self.epistemic_std)) if self.epistemic_std is not None else None
                },
                'aleatoric': {
                    'description': 'Data noise uncertainty (MSE-based)',
                    'available': self.aleatoric_stats is not None,
                    **(self.aleatoric_stats if self.aleatoric_stats else {})
                },
                'conformal': {
                    'description': 'Distribution-free prediction intervals',
                    'available': self.q_ is not None,
                    'alpha': self.alpha,
                    'quantile': self.q_
                }
            }
        }
        
        # Add coverage analysis
        if self.q_ is not None:
            report['coverage_analysis'] = self.compute_coverage(y_true, y_pred)
        
        # Add correlation analysis
        report['correlation_analysis'] = self.analyze_uncertainty_correlation(y_true, y_pred)
        
        # Add prediction quality metrics
        residuals = y_true - y_pred
        report['prediction_quality'] = {
            'mse': float(np.mean(residuals ** 2)),
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'mae': float(np.mean(np.abs(residuals))),
            'r2_score': float(1 - (np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)))
        }
        
        return report


