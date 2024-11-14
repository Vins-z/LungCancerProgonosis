"""
# Lung Cancer Prediction Application

A production-ready application for predicting lung cancer using machine learning algorithms.

## Features
- Random Forest and Branch & Bound classification algorithms
- Interactive GUI for data upload and model selection
- Comprehensive visualization capabilities
- Robust error handling and logging
- Production-grade code structure

## Installation
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`

## Usage
1. Run the application: `python main.py`
2. Upload your dataset (CSV format)
3. Select a prediction model
4. View results and generated visualizations

## Project Structure
```
lung_cancer_prediction/
│
├── config/
│   └── config.py         # Configuration settings
│
├── src/
│   ├── models/
│   │   └── algorithms.py # Model implementations
│   ├── data/
│   │   └── data_processor.py # Data handling
│   ├── visualization/
│   │   └── visualizer.py # Plotting functions
│   └── gui/
│       └── app.py        # GUI implementation
│
├── output/              # Generated files
├── main.py             # Application entry point
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```
## License
MIT License

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request