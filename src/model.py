import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add the MPIIGaze demo to Python path
mpiigaze_path = Path(__file__).parent.parent / "pytorch_mpiigaze_demo"
sys.path.append(str(mpiigaze_path))

try:
    from ptgaze.models import create_model
    from ptgaze.transforms import create_transform
    from ptgaze.gaze_estimator import GazeEstimator
    from omegaconf import DictConfig
    MPIIGAZE_AVAILABLE = True
except ImportError:
    MPIIGAZE_AVAILABLE = False
    print("Warning: MPIIGaze not available. Install dependencies or check path.")


class TinyMLP(nn.Module):
    """Simple MLP for tabular gaze features"""
    def __init__(self, d_in: int, d_h: int = 96, p_drop: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d_h, d_h), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d_h, 1), nn.Sigmoid()   # outputs in [0,1]
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class MultimodalModel(nn.Module):
    """Multimodal model combining tabular features with eye images"""
    
    def __init__(self, 
                 tabular_feature_dim=5,
                 gaze_feature_dim=None,
                 hidden_dim=128,
                 dropout=0.3,
                 freeze_gaze=True):
        super().__init__()
        
        # Gaze feature extractor (MPIIGaze)
        self.gaze_extractor = None
        if MPIIGAZE_AVAILABLE:
            try:
                self.gaze_extractor = self._create_gaze_extractor()
                if freeze_gaze:
                    for param in self.gaze_extractor.parameters():
                        param.requires_grad = False
            except Exception as e:
                print(f"Warning: Could not create gaze extractor: {e}")
                self.gaze_extractor = None
        
        # Determine gaze feature dimension
        if gaze_feature_dim is None:
            gaze_feature_dim = 64 if self.gaze_extractor else 0
        
        # Combined feature dimension
        total_feature_dim = tabular_feature_dim + gaze_feature_dim
        
        # Engagement prediction head
        self.engagement_head = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.tabular_feature_dim = tabular_feature_dim
        self.gaze_feature_dim = gaze_feature_dim
    
    def _create_gaze_extractor(self):
        """Create MPIIGaze feature extractor"""
        # Simplified gaze extractor for production
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 16, 64)
        )
    
    def forward(self, tabular_features, eye_images=None):
        """Forward pass combining tabular and visual features"""
        if self.gaze_extractor is not None and eye_images is not None:
            # Extract gaze features from eye images
            gaze_features = self.gaze_extractor(eye_images)
            # Combine tabular and gaze features
            combined_features = torch.cat([tabular_features, gaze_features], dim=1)
        else:
            # Use only tabular features
            combined_features = tabular_features
        
        # Predict engagement
        engagement_scores = self.engagement_head(combined_features)
        return engagement_scores.squeeze(-1)


def create_model(model_type="tiny", **kwargs):
    """Factory function to create models"""
    if model_type == "multimodal" and MPIIGAZE_AVAILABLE:
        return MultimodalModel(**kwargs)
    else:
        # Map multimodal parameters to TinyMLP parameters
        tiny_kwargs = {
            'd_in': kwargs.get('tabular_feature_dim', kwargs.get('d_in', 5)),
            'd_h': kwargs.get('hidden_dim', kwargs.get('d_h', 96)),
            'p_drop': kwargs.get('dropout', kwargs.get('p_drop', 0.3))
        }
        return TinyMLP(**tiny_kwargs)


def gpu_status_str() -> str:
    """GPU status string for logging"""
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024**3)
            res = torch.cuda.memory_reserved(0) / (1024**3)
            alloc = torch.cuda.memory_allocated(0) / (1024**3)
            return f"CUDA: {name} | Total {total:.1f} GB | Reserved {res:.2f} GB | Alloc {alloc:.2f} GB"
        except Exception as e:
            return f"CUDA available; memory query failed: {e}"
    return "CUDA not available; using CPU."
