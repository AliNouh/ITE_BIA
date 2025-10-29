from setuptools import setup, find_packages

setup(
    name="ite_bia",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=1.0',
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'streamlit>=1.15.0',
        'plotly>=5.3.0',
        'joblib>=1.1.0',
        'pytest>=7.0.0'
    ]
)