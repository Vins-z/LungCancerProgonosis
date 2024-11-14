from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                            QFileDialog, QLabel, QRadioButton, QButtonGroup, 
                            QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt
import sys
import logging
from config import Config
from data_processor import DataProcessor
from visualizer import Visualizer

logger = logging.getLogger(__name__)

class LungCancerPredictionApp(QWidget):
    """Main application GUI."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.data_processor = DataProcessor(config)
        self.visualizer = Visualizer(config)
        self.setup_ui()
        
    def setup_ui(self) -> None:
        """Initialize the user interface."""
        try:
            self.setWindowTitle('Lung Cancer Prediction')
            self.setGeometry(100, 100, 500, 400)
            
            layout = QVBoxLayout()
            
            # Add UI components
            self.file_label = QLabel('No file loaded')
            self.upload_button = QPushButton('Upload Dataset')
            self.upload_button.clicked.connect(self.upload_file)
            
            self.model_label = QLabel('Select a Model:')
            self.rf_button = QRadioButton('Random Forest')
            self.bb_button = QRadioButton('Branch and Bound')
            self.rf_button.setChecked(True)
            
            self.model_group = QButtonGroup()
            self.model_group.addButton(self.rf_button)
            self.model_group.addButton(self.bb_button)
            
            self.predict_button = QPushButton('Run Prediction')
            self.predict_button.clicked.connect(self.run_prediction)
            
            self.result_label = QTextEdit()
            self.result_label.setReadOnly(True)
            
            # Add widgets to layout
            for widget in [self.file_label, self.upload_button, self.model_label,
                         self.rf_button, self.bb_button, self.predict_button,
                         self.result_label]:
                layout.addWidget(widget)
            
            self.setLayout(layout)
            
        except Exception as e:
            logger.error(f"Error setting up UI: {str(e)}")
            raise
    
    def upload_file(self) -> None:
        """Handle file upload."""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self, 'Open CSV File', '', 
                'CSV Files (*.csv);;All Files (*)', 
                options=options
            )
            
            if file_path:
                self.data = pd.read_csv(file_path)
                self.file_label.setText(f'File Loaded: {file_path}')
                logger.info(f"File loaded: {file_path}")
                
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error uploading file: {str(e)}")
    
    def run_prediction(self) -> None:
        """Execute the selected model prediction."""
        try:
            if not hasattr(self, 'data'):
                raise ValueError("No data loaded. Please upload a dataset first.")
            
            # Preprocess data
            X, y = self.data_processor.preprocess_data(self.data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.TEST_SIZE, 
                random_state=self.config.RANDOM_STATE
            )
            
            # Train and evaluate model
            if self.rf_button.isChecked():
                model = RandomForestClassifier(random_state=self.config.RANDOM_STATE)
                self._run_random_forest(model, X_train, X_test, y_train, y_test)
            else:
                model = BranchAndBoundClassifier(max_depth=self.config.MAX_DEPTH)
                self._run_branch_and_bound(model, X_train, X_test, y_train, y_test)
                
        except Exception as e:
            logger.error(f"Error running prediction: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error running prediction: {str(e)}")
    
    def _run_random_forest(self, model, X_train, X_test, y_train, y_test) -> None:
        """Execute Random Forest prediction."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Generate visualizations and reports
        self.visualizer.plot_feature_importance(
            model, 
            self.data.drop('LUNG_CANCER', axis=1).columns
        )
        
        report = classification_report(y_test, y_pred)
        self.result_label.setText(
            f'Random Forest Results:\n\n'
            f'Classification Report:\n{report}\n\n'
            f'Visualizations saved to {self.config.OUTPUT_DIR}'
        )
    
    def _run_branch_and_bound(self, model, X_train, X_test, y_train, y_test) -> None:
        """Execute Branch and Bound prediction."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        report = classification_report(y_test, y_pred)
        self.result_label.setText(
            f'Branch and Bound Results:\n\n'
            f'Classification Report:\n{report}'
        )