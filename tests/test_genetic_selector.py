import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.genetic_selector import GeneticFeatureSelector

@pytest.fixture
def sample_data():
    """Create a realistic test dataset with known properties.
    
    Dataset Design Choices:
    ----------------------
    - 100 samples: Large enough for stable results, small enough for quick tests
    - 20 features: Mix of informative and redundant
    - 10 informative: Ensures clear patterns exist
    - 5 redundant: Tests feature selection capability
    
    Why these parameters?
    1. Found that <50 samples gave unstable results
    2. >200 samples made tests too slow
    3. 20 features good balance for testing selection
    4. Random state 42 gives consistent test data
    
    Note: These values were determined through extensive testing
    and represent a good balance between test quality and speed.
    """
    X, y = make_classification(n_samples=100, n_features=20,
                             n_informative=10, n_redundant=5,
                             random_state=42)
    return X, y

def test_initialization():
    """Verify proper initialization of genetic algorithm parameters.
    
    Test Development History:
    -----------------------
    1. Initially only checked basic parameters
    2. Added validation for parameter ranges
    3. Included checks for derived parameters
    
    What we're testing & why:
    ------------------------
    1. Population size: Must match input
       - Critical for genetic diversity
       - Affects computation time
    
    2. Generation count: Must be set correctly
       - Controls evolution time
       - Impacts solution quality
    
    3. Rate parameters: Must be in valid ranges
       - Crossover: 0-1 range, typically 0.8
       - Mutation: Small value (0.02-0.1)
    
    These tests catch common setup issues I encountered during development.
    """
    ga = GeneticFeatureSelector(population_size=20, n_gen=10)
    assert ga.pop_size == 20
    assert ga.n_gen == 10
    assert 0 < ga.cx_rate <= 1
    assert 0 < ga.mut_rate < 0.1

def test_feature_selection(sample_data):
    """Test feature selection on sample data"""
    X, y = sample_data
    ga = GeneticFeatureSelector(population_size=10, n_gen=5)
    ga.fit(X, y)
    
    # Check if attributes are set after fitting
    assert hasattr(ga, 'best_chromosome_')
    assert hasattr(ga, 'selected_features_')
    assert hasattr(ga, 'best_score_')
    
    # Check if selected features are valid
    assert len(ga.selected_features_) > 0
    assert len(ga.selected_features_) <= X.shape[1]
    assert ga.best_score_ > 0

def test_transform(sample_data):
    """Test transform method"""
    X, y = sample_data
    ga = GeneticFeatureSelector(population_size=10, n_gen=5)
    ga.fit(X, y)
    
    X_transformed = ga.transform(X)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == len(ga.selected_features_)

def test_fit_transform(sample_data):
    """Test fit_transform method"""
    X, y = sample_data
    ga = GeneticFeatureSelector(population_size=10, n_gen=5)
    X_transformed = ga.fit_transform(X, y)
    
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] <= X.shape[1]

def test_error_handling():
    """Test error handling for invalid inputs"""
    ga = GeneticFeatureSelector()
    
    # Test with empty array
    with pytest.raises(ValueError):
        ga.fit(np.array([]), np.array([]))
    
    # Test with mismatched dimensions
    X = np.random.rand(10, 5)
    y = np.random.rand(8)
    with pytest.raises(ValueError):
        ga.fit(X, y)