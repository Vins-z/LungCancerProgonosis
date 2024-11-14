import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Visualizer:
    """Handle all visualization tasks."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def save_plot(self, fig: plt.Figure, filename: str) -> None:
        """Save plot to output directory."""
        try:
            output_path = self.config.OUTPUT_DIR / filename
            fig.savefig(output_path)
            plt.close(fig)
            logger.info(f"Plot saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving plot {filename}: {str(e)}")
            raise

    def plot_feature_importance(self, model: RandomForestClassifier, 
                              feature_names: List[str]) -> None:
        """Plot feature importance for Random Forest model."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            ax.bar(range(len(importances)), importances[indices])
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
            ax.set_title("Feature Importances")
            plt.tight_layout()
            
            self.save_plot(fig, 'feature_importance.png')
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
