"""
Pure unit tests for genetic feature selection without sklearn dependencies.
"""
import pytest
import numpy as np
from src.pure_genetic_selector import GeneticFeatureSelector
from src.pure_compare_methods import run_comparisons

def generate_test_data(n_samples=100, n_features=20, random_state=42):
    """Generate synthetic data for testing."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    # Create target based on first two features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def test_ga_initialization():
    """Test GeneticFeatureSelector initialization."""
    ga = GeneticFeatureSelector(population_size=20, n_gen=10)
    assert ga.pop_size == 20
    assert ga.n_gen == 10
    assert 0 < ga.cx_rate <= 1
    assert 0 < ga.mut_rate < 0.1

def test_feature_selection():
    """Test feature selection on synthetic data."""
    X, y = generate_test_data()
    ga = GeneticFeatureSelector(population_size=10, n_gen=5)
    ga.fit(X, y)
    
    # Check attributes are set
    assert hasattr(ga, 'best_chromosome_')
    assert hasattr(ga, 'selected_features_')
    assert hasattr(ga, 'best_score_')
    
    # Check selected features
    assert len(ga.selected_features_) > 0
    assert len(ga.selected_features_) <= X.shape[1]
    assert ga.best_score_ > 0

def test_transform():
    """Test transform method."""
    X, y = generate_test_data()
    ga = GeneticFeatureSelector(population_size=10, n_gen=5)
    ga.fit(X, y)
    
    X_transformed = ga.transform(X)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == len(ga.selected_features_)

def test_comparisons():
    """Test comparison methods."""
    X, y = generate_test_data()
    results = run_comparisons(X, y, k=5)
    
    # Check results structure
    assert 'GA' in results
    assert 'Correlation' in results
    assert 'Variance' in results
    assert 'Random' in results
    
    # Check metrics
    for method, metrics in results.items():
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 'n_features' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert metrics['n_features'] > 0

def test_error_handling():
    """Test error handling for invalid inputs."""
    ga = GeneticFeatureSelector()
    
    # Test empty arrays
    with pytest.raises(ValueError):
        ga.fit(np.array([]), np.array([]))
    
    # Test mismatched dimensions
    with pytest.raises(ValueError):
        X = np.random.rand(10, 5)
        y = np.random.rand(8)
        ga.fit(X, y)

def test_chromosome_validation():
    """Test chromosome constraints are maintained."""
    X, y = generate_test_data(n_samples=50, n_features=10)
    ga = GeneticFeatureSelector(population_size=5, n_gen=3)
    ga.fit(X, y)
    
    # Check each chromosome has at least one feature
    pop = ga._init_population(X.shape[1])
    for chrom in pop:
        assert np.sum(chrom) > 0

def test_consistent_results():
    """Test consistency with fixed random state."""
    X, y = generate_test_data()
    
    ga1 = GeneticFeatureSelector(random_state=42)
    ga2 = GeneticFeatureSelector(random_state=42)
    
    ga1.fit(X, y)
    ga2.fit(X, y)
    
    assert np.array_equal(ga1.selected_features_, ga2.selected_features_)
    assert ga1.best_score_ == ga2.best_score_