# Feature Selection using Genetic Algorithm (GA) — BIA601 (RAFD) Project Report

## 1. Executive Summary
The objective of this project is to implement a genetic algorithm for feature selection (GA) and compare it with traditional and statistical methods (Lasso, PCA, Chi-square, ANOVA), providing a software product that allows uploading CSV files and running analysis easily. The project includes a Streamlit interface for running analysis and outputting results.

## 2. Requirements Specification
- Support for numerical data (preferably tabular). CSV format upload is supported.
- No need for deep learning models.
- Use fitness function to measure feature quality.
- Feature pruning and dimensionality reduction.
- Comparison with Lasso, PCA, and statistical tests like Chi-square.
- Clear documentation and explanation.

## 3. Dataset Used
The project includes ready-made examples (Breast Cancer from sklearn). Users can upload CSV files containing a target column.

## 4. Solution Methodology
### 4.1 Genetic Algorithm (GA)
- Chromosome representation: Binary mask (0/1) for each feature.
- Fitness function: Average RandomForest accuracy via cross-validation.
- Default parameters: population_size=30, generations=25, crossover=0.8, mutation=0.02.
- Selection mechanisms: Tournament selection, Single-point crossover, Bit-flip mutation.

### 4.2 Comparisons
- LassoCV for selecting features with non-zero weights.
- SelectKBest(chi2) for chi-square test.
- SelectKBest(f_classif) for ANOVA test.
- PCA for dimensionality reduction to first k components.

## 5. Results
When running `web/app.py` or `src/compare_methods.py`, a table is produced comparing each method in: accuracy, F1 (f1_macro), and number of features.

## 6. How to Run
1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. To run web interface:
   ```
   streamlit run web/app.py
   ```
3. To run comparisons via command line:
   ```
   python -m src.compare_methods
   ```

## 7. Delivery Notes
- English documentation and README file included.
- Project organized in src, web, docs, results, examples folders.
- GA can be improved by adding multi-objective fitness (accuracy vs #features) and parameter tuning.

## 8. Future Suggestions
- Support for text data (feature engineering for text).
- Add automatically generated PDF reports.
- Complete unit tests to ensure module integrity.

## 9. Technical Implementation Details
### 9.1 Genetic Algorithm Components
- **Population Initialization**: Random binary chromosomes with at least one feature selected
- **Fitness Evaluation**: Cross-validation accuracy using RandomForest classifier
- **Selection**: Tournament selection with tournament size of 3
- **Crossover**: Single-point crossover with 80% probability
- **Mutation**: Bit-flip mutation with 2% probability per gene

### 9.2 Comparison Methods
- **Lasso Regularization**: L1 regularization for automatic feature selection
- **Principal Component Analysis**: Linear dimensionality reduction
- **Chi-square Test**: Statistical test for categorical target variables
- **ANOVA F-test**: Statistical test for continuous features

## 10. Performance Metrics
All methods are evaluated using:
- **Accuracy**: Classification accuracy via 5-fold cross-validation
- **F1 Score**: Macro-averaged F1 score for balanced evaluation
- **Feature Count**: Number of selected/transformed features

## 11. Code Structure
- `src/genetic_selector.py`: Main GA implementation
- `src/compare_methods.py`: Comparison framework and evaluation
- `web/app.py`: Streamlit web interface
- `demo_simple.py`: Simplified standalone demo
- `docs/`: Documentation and reports
- `examples/`: Sample datasets
