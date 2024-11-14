from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from config import Config  

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handle data preprocessing and validation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate input data structure and content."""
        required_columns = {'LUNG_CANCER'}  # Add all required columns
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if df.empty:
            raise ValueError("DataFrame is empty")
            
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for model training."""
        try:
            self.validate_data(df)
            df_processed = df.copy()
            
            # Convert categorical variables to numerical
            for column in df_processed.columns:
                if df_processed[column].dtype == 'object':
                    self.label_encoders[column] = LabelEncoder()
                    df_processed[column] = self.label_encoders[column].fit_transform(df_processed[column])
            
            X = df_processed.drop('LUNG_CANCER', axis=1).values
            y = df_processed['LUNG_CANCER'].values
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise