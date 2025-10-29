# Feature Selection using Genetic Algorithm

## 🧬 Project Overview
This project implements an intelligent feature selection system using Genetic Algorithms. It aims to optimize model performance while reducing the number of features used in machine learning tasks.

## 🎯 Key Features
- Genetic Algorithm-based feature selection
- Comparison with traditional methods (Lasso, Chi-Square, ANOVA, PCA)
- Interactive web interface using Streamlit
- Support for custom datasets
- Comprehensive performance analysis and visualization
- PDF report generation

## 📊 Supported Methods
1. **Genetic Algorithm**
   - Customizable population size and generations
   - Adaptive mutation and crossover
   - Fitness-based selection

2. **Traditional Methods**
   - Lasso Regularization
   - Chi-Square Feature Selection
   - ANOVA F-test
   - Principal Component Analysis (PCA)

## 🚀 Quick Start

### Prerequisites
```bash
python -version >= 3.9
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/AliNouh/ITE_BIA.git
cd ITE_BIA
```

2. Install minimal requirements:
```bash
pip install -r requirements-minimal.txt
```

### Running the Application
```bash
cd web
streamlit run app.py
```

## 📁 Project Structure
```
ITE_BIA/
├── src/                    # Core implementation
│   ├── genetic_selector.py # Genetic algorithm implementation
│   └── compare_methods.py  # Traditional methods implementation
├── web/                    # Web interface
│   ├── app.py             # Main Streamlit application
│   └── helpers.py         # UI helper functions
├── tests/                 # Unit tests
├── examples/              # Example datasets
└── tools/                # Utility scripts
```

## 💡 Usage Examples

### Using Sample Datasets
1. Launch the web application
2. Select "Sample Dataset" option
3. Choose between Breast Cancer or Kidney Disease datasets
4. Adjust parameters as needed
5. Click "Run Analysis"

### Using Custom Datasets
1. Prepare your CSV file with features and target column
2. Select "Upload Custom File" option
3. Choose your CSV file
4. Select the target column
5. Configure analysis parameters
6. Run the analysis

## 📊 Performance Metrics
The system evaluates feature selection using:
- Accuracy
- F1 Score
- Feature Reduction Rate
- Processing Time

## 📈 Results Visualization
- Performance comparison charts
- Feature importance plots
- Interactive results analysis
- Downloadable PDF reports

## 🧪 Testing
Run the test suite:
```bash
python -m pytest tests/
```

## 👥 Contributors
- Ali Nouh
- Abd
- Sumer
- Hamza
- Ahmed
- Batoul
- Saeed
- Marla

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments
Special thanks to the BIA601 team and all contributors who provided valuable feedback during development.
