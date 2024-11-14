import logging
from config import Config
from PyQt5.QtWidgets import QApplication
from lung_cancer_prediction_app import LungCancerPredictionApp  # Import the class
def setup_logging(config: Config) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOGGING_FORMAT,
        handlers=[
            logging.FileHandler(config.OUTPUT_DIR / "app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set logging levels for specific modules
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")

def check_dependencies() -> None:
    """Verify all required dependencies are available."""
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'PyQt5': 'PyQt5',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        raise ImportError(
            f"Missing required packages. Please install:\n"
            f"pip install {' '.join(missing_packages)}"
        )

def setup_exception_handling(app: QApplication) -> None:
    """Configure global exception handling."""
    def exception_hook(exctype, value, traceback):
        """Handle uncaught exceptions."""
        logger = logging.getLogger(__name__)
        logger.error("Uncaught exception:", exc_info=(exctype, value, traceback))
        sys.__excepthook__(exctype, value, traceback)  # Call the default handler
    
    sys.excepthook = exception_hook

def create_required_directories(config: Config) -> None:
    """Create necessary directories for the application."""
    try:
        directories = [
            config.OUTPUT_DIR,
            config.OUTPUT_DIR / "plots",
            config.OUTPUT_DIR / "models",
            config.OUTPUT_DIR / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger = logging.getLogger(__name__)
        logger.info("Required directories created successfully")
        
    except Exception as e:
        raise RuntimeError(f"Failed to create required directories: {str(e)}")

def main() -> None:
    """Main application entry point."""
    try:
        # Initialize configuration
        config = Config()
        
        # Setup application infrastructure
        create_required_directories(config)
        setup_logging(config)
        check_dependencies()
        
        # Initialize Qt application
        app = QApplication(sys.argv)
        setup_exception_handling(app)
        
        # Create and show main window
        logger = logging.getLogger(__name__)
        logger.info("Starting Lung Cancer Prediction application")
        
        window = LungCancerPredictionApp(config)
        window.show()
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()