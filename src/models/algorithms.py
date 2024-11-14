import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Union, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class TreeNode:
    """Binary decision tree node."""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional[Union['TreeNode', int]] = None
    right: Optional[Union['TreeNode', int]] = None
    prediction: Optional[int] = None

class BranchAndBoundClassifier(BaseEstimator, ClassifierMixin):
    """Custom Branch and Bound classifier implementation."""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.tree = None
    
    @staticmethod
    def gini_impurity(y: np.ndarray) -> float:
        m = len(y)
        if m == 0:
            return 0
        prob = np.bincount(y) / m
        return 1 - np.sum(prob ** 2)
    
    def information_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        m = len(y)
        if m == 0:
            return 0
        return (self.gini_impurity(y) - 
                (len(y_left) / m) * self.gini_impurity(y_left) - 
                (len(y_right) / m) * self.gini_impurity(y_right))
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        best_feature, best_threshold, best_gain = None, None, -np.inf
        
        try:
            for feature in range(X.shape[1]):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    left_mask = X[:, feature] <= threshold
                    right_mask = X[:, feature] > threshold
                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue
                    
                    gain = self.information_gain(y, y[left_mask], y[right_mask])
                    if gain > best_gain:
                        best_feature, best_threshold, best_gain = feature, threshold, gain
                        
        except Exception as e:
            logger.error(f"Error finding best split: {str(e)}")
            raise
            
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        node = TreeNode()
        
        if depth == self.max_depth or self.gini_impurity(y) == 0:
            node.prediction = int(np.argmax(np.bincount(y)))
            return node
            
        feature, threshold = self._find_best_split(X, y)
        
        if feature is None:
            node.prediction = int(np.argmax(np.bincount(y)))
            return node
            
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BranchAndBoundClassifier':
        """Build the decision tree."""
        try:
            self.tree = self._build_tree(X, y)
            return self
        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise
    
    def _predict_single(self, x: np.ndarray, node: TreeNode) -> int:
        """Predict for a single sample."""
        if node.prediction is not None:
            return node.prediction
            
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        if self.tree is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        try:
            return np.array([self._predict_single(x, self.tree) for x in X])
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise