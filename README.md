# Feature Selection using Genetic Algorithm (GA) — BIA601 (RAFD)

## Project Structure:
- **src/**: Project code (GA + method comparison)
- **web/**: Streamlit application for CSV upload and analysis
- **docs/**: Documentation and reports
- **results/**: Directory for saving output results
- **examples/**: Sample data files

## Quick Start:
1. Install requirements: `pip install -r requirements.txt`
2. Run web interface: `streamlit run web/app.py`
3. Or run comparisons directly: `python -m src.compare_methods`

**Note**: Ensure the target column in uploaded CSV files is named 'target' or select it from the interface.
