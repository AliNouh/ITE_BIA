"""
Simple feature selection comparison methods without sklearn dependencies.
"""
import numpy as np
from src.pure_genetic_selector import GeneticFeatureSelector

def simple_correlation_selection(X, y, k=None):
    """Select features based on correlation with target."""
    if k is None:
        k = X.shape[1] // 4
    
    correlations = np.array([
        abs(np.corrcoef(X[:, i], y)[0, 1])
        for i in range(X.shape[1])
    ])
    
    return np.argsort(correlations)[-k:]

def simple_variance_selection(X, y, k=None):
    """Select features with highest variance."""
    if k is None:
        k = X.shape[1] // 4
    
    variances = np.var(X, axis=0)
    return np.argsort(variances)[-k:]

def simple_random_selection(X, y, k=None):
    """Random feature selection as baseline."""
    if k is None:
        k = X.shape[1] // 4
    
    return np.random.choice(X.shape[1], size=k, replace=False)

def evaluate_feature_subset(X, y, feature_idx):
    """Evaluate feature subset using simple KNN classifier."""
    if len(feature_idx) == 0:
        return {'accuracy': 0.0, 'f1': 0.0, 'n_features': 0}
    
    X_sel = X[:, feature_idx]
    
    # K-fold cross validation
    k = 5
    fold_size = len(y) // k
    accuracies = []
    f1_scores = []
    
    for i in range(k):
        # Split data
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else len(y)
        
        X_train = np.concatenate([X_sel[:test_start], X_sel[test_end:]])
        y_train = np.concatenate([y[:test_start], y[test_end:]])
        X_test = X_sel[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Simple KNN prediction
        predictions = []
        for x_t in X_test:
            distances = np.sqrt(np.sum((X_train - x_t) ** 2, axis=1))
            nearest = np.argsort(distances)[:3]
            pred = np.bincount(y_train[nearest]).argmax()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y_test)
        
        # Calculate F1 score
        tp = np.sum((predictions == 1) & (y_test == 1))
        fp = np.sum((predictions == 1) & (y_test == 0))
        fn = np.sum((predictions == 0) & (y_test == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracies.append(accuracy)
        f1_scores.append(f1)
    
    return {
        'accuracy': float(np.mean(accuracies)),
        'f1': float(np.mean(f1_scores)),
        'n_features': int(len(feature_idx))
    }

def run_comparisons(X, y, k=None, random_state=42):
    """Run all feature selection methods and compare results."""
    np.random.seed(random_state)
    
    if k is None:
        k = max(1, X.shape[1] // 4)
    
    results = {}
    
    # Genetic Algorithm
    ga = GeneticFeatureSelector(
        population_size=30,
        n_gen=25,
        random_state=random_state
    )
    ga.fit(X, y)
    results['GA'] = evaluate_feature_subset(X, y, ga.selected_features_)
    
    # Correlation-based
    corr_idx = simple_correlation_selection(X, y, k)
    results['Correlation'] = evaluate_feature_subset(X, y, corr_idx)
    
    # Variance-based
    var_idx = simple_variance_selection(X, y, k)
    results['Variance'] = evaluate_feature_subset(X, y, var_idx)
    
    # Random (baseline)
    rand_idx = simple_random_selection(X, y, k)
    results['Random'] = evaluate_feature_subset(X, y, rand_idx)
    
    return results

if __name__ == "__main__":
    # Generate sample data
    n_samples, n_features = 100, 20
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple binary classification
    
    results = run_comparisons(X, y, k=5)
    
    print("\nResults:")
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Features: {metrics['n_features']}")