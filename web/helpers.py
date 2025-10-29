"""
Helper functions for running analysis in ITE_BIA application
"""

# Core Python libraries
import json
import time
from pathlib import Path

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Machine Learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

def run_analysis(X_scaled, y, k, n_gen, pop_size, n_estimators, random_state):
    """
    Run comparative analysis between different algorithms
    
    Args:
        X_scaled: Processed and scaled data
        y: Target variable
        k: Number of desired features
        n_gen: Number of generations in genetic algorithm
        pop_size: Population size in genetic algorithm
        n_estimators: Number of trees in RandomForest
        random_state: Random seed for reproducibility
    
    Returns:
        dict: Results of algorithm comparisons
    """
    print(f"🧬 Starting comparative analysis...")
    print(f"   - Number of samples: {X_scaled.shape[0]}")
    print(f"   - Original features: {X_scaled.shape[1]}")
    print(f"   - Desired features: {k}")
    print(f"   - Number of generations: {n_gen}")
    print(f"   - Population size: {pop_size}")
    
    results = {}
    
    # 1. Genetic Algorithm
    print(f"\n🧬 Running Genetic Algorithm...")
    try:
        from src.genetic_selector import GeneticSelector
        selector = GeneticSelector(
            n_features=k,
            n_gen=n_gen,
            pop_size=pop_size,
            random_state=random_state
        )
        selector.fit(X_scaled, y)
        
        # Get best features
        selected_features = selector.get_selected_features()
        
        # Train the model
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(selected_features, y)
        
        # Evaluation
        y_pred = clf.predict(selected_features)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        
        results['Genetic Algorithm'] = {
            'accuracy': accuracy,
            'f1': f1,
            'n_features': k,
            'selector': selector,
            'selected_features': selected_features
        }
        
        print(f"✅ Genetic Algorithm completed")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - F1 Score: {f1:.4f}")
        
    except Exception as e:
        print(f"❌ Error in Genetic Algorithm: {e}")
        results['Genetic Algorithm'] = {
            'accuracy': 0.0,
            'f1': 0.0,
            'n_features': k,
            'error': str(e)
        }
    
    # 2. Lasso
    print(f"\n📊 Running Lasso...")
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.feature_selection import SelectFromModel
        
        lasso = LassoCV(cv=5, random_state=random_state, max_iter=1000)
        lasso.fit(X_scaled, y)
        
        # Select important features
        sfm = SelectFromModel(lasso, prefit=True)
        selected_features = sfm.transform(X_scaled)
        
        # تدريب النموذج
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(selected_features, y)
        
        # التقييم
        y_pred = clf.predict(selected_features)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        
        results['Lasso'] = {
            'accuracy': accuracy,
            'f1': f1,
            'n_features': selected_features.shape[1],
            'selected_features': selected_features
        }
        
        print(f"✅ Lasso completed")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - F1 Score: {f1:.4f}")
        print(f"   - Selected features: {selected_features.shape[1]}")
        
    except Exception as e:
        print(f"❌ Error in Lasso: {e}")
        results['Lasso'] = {
            'accuracy': 0.0,
            'f1': 0.0,
            'n_features': k,
            'error': str(e)
        }
    
    # 3. Chi-Square
    print(f"\n📊 Running Chi-Square...")
    try:
        from sklearn.feature_selection import SelectKBest, chi2
        
        # Ensure no negative values for Chi2
        X_non_negative = np.abs(X_scaled)
        
        chi2_selector = SelectKBest(chi2, k=min(k, X_scaled.shape[1]))
        selected_features = chi2_selector.fit_transform(X_non_negative, y)
        
        # تدريب النموذج
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(selected_features, y)
        
        # التقييم
        y_pred = clf.predict(selected_features)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        
        results['Chi-Square'] = {
            'accuracy': accuracy,
            'f1': f1,
            'n_features': selected_features.shape[1],
            'selected_features': selected_features,
            'scores': chi2_selector.scores_
        }
        
        print(f"✅ Chi-Square completed")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - F1 Score: {f1:.4f}")
        print(f"   - Selected features: {selected_features.shape[1]}")
        
    except Exception as e:
        print(f"❌ Error in Chi-Square: {e}")
        results['Chi-Square'] = {
            'accuracy': 0.0,
            'f1': 0.0,
            'n_features': k,
            'error': str(e)
        }
    
    # 4. ANOVA
    print(f"\n📊 Running ANOVA...")
    try:
        from sklearn.feature_selection import SelectKBest, f_classif
        
        anova_selector = SelectKBest(f_classif, k=min(k, X_scaled.shape[1]))
        selected_features = anova_selector.fit_transform(X_scaled, y)
        
        # تدريب النموذج
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(selected_features, y)
        
        # التقييم
        y_pred = clf.predict(selected_features)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        
        results['ANOVA'] = {
            'accuracy': accuracy,
            'f1': f1,
            'n_features': selected_features.shape[1],
            'selected_features': selected_features,
            'scores': anova_selector.scores_
        }
        
        print(f"✅ ANOVA completed")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - F1 Score: {f1:.4f}")
        print(f"   - Selected features: {selected_features.shape[1]}")
        
    except Exception as e:
        print(f"❌ Error in ANOVA: {e}")
        results['ANOVA'] = {
            'accuracy': 0.0,
            'f1': 0.0,
            'n_features': k,
            'error': str(e)
        }
    
    # 5. PCA
    print(f"\n📊 Running PCA...")
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # PCA لاختيار أفضل المكونات
        pca = PCA(n_components=min(k, X_scaled.shape[1]))
        selected_features = pca.fit_transform(X_scaled)
        
        # تدريب النموذج
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(selected_features, y)
        
        # التقييم
        y_pred = clf.predict(selected_features)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        
        results['PCA'] = {
            'accuracy': accuracy,
            'f1': f1,
            'n_features': selected_features.shape[1],
            'selected_features': selected_features,
            'explained_variance': pca.explained_variance_ratio_.sum(),
            'components': pca.components_
        }
        
        print(f"✅ PCA completed")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - F1 Score: {f1:.4f}")
        print(f"   - Selected components: {selected_features.shape[1]}")
        print(f"   - Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        
    except Exception as e:
        print(f"❌ Error in PCA: {e}")
        results['PCA'] = {
            'accuracy': 0.0,
            'f1': 0.0,
            'n_features': k,
            'error': str(e)
        }
    
    print(f"\n🎉 Comparative analysis completed!")
    print(f"Compared {len(results)} different algorithms")
    
    return results

def display_results(results, original_features_count):
    """
    Display comparative analysis results
    
    Args:
        results: Comparison results
        original_features_count: Number of original features
    """
    print(f"\n📊 Comparative Analysis Results:")
    print(f"Number of original features: {original_features_count}")
    print("-" * 80)
    
    # Sort results by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (method, metrics) in enumerate(sorted_results, 1):
        accuracy = metrics['accuracy']
        f1 = metrics['f1']
        n_features = metrics['n_features']
        
        # Calculate feature reduction rate
        reduction_rate = (1 - n_features / original_features_count) * 100
        
        print(f"{i}. {method}:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - F1 Score: {f1:.4f}")
        print(f"   - Number of Features: {n_features}")
        print(f"   - Feature Reduction: {reduction_rate:.1f}%")
        
        if 'error' in metrics:
            print(f"   - خطأ: {metrics['error']}")
        
        print()

def estimate_time(n_samples, n_features, n_gen, pop_size, performance_mode):
    """
    Estimate analysis time
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_gen: Number of generations
        pop_size: Population size
        performance_mode: Performance mode
    
    Returns:
        float: Estimated time in minutes
    """
    # Performance mode adjustment factor
    mode_multiplier = {
        "Very Fast ⚡": 0.5,
        "Fast 🚀": 0.8,
        "Balanced ⚖️": 1.0,
        "Accurate 🎯": 1.5
    }
    
    # Base equation
    base_time = (n_samples * n_features * n_gen * pop_size) / (1e6)
    
    # Adjust according to performance mode
    adjusted_time = base_time * mode_multiplier.get(performance_mode, 1.0)
    
    # Add overhead time
    overhead_time = 0.5  # Fixed overhead time
    
    return adjusted_time + overhead_time

def save_results(results, filename="results.json"):
    """
    Save results to a JSON file
    
    Args:
        results: Results to save
        filename: File name
    """
    try:
        # Convert results to serializable JSON format
        serializable_results = {}
        for method, metrics in results.items():
            serializable_metrics = {}
            for key, value in metrics.items():
                if key in ['accuracy', 'f1', 'n_features']:
                    serializable_metrics[key] = value
                elif key == 'error':
                    serializable_metrics[key] = str(value)
                else:
                    # Store other data as references
                    serializable_metrics[key] = f"<{type(value).__name__}>"
            
            serializable_results[method] = serializable_metrics
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ تم حفظ النتائج في {filename}")
        
    except Exception as e:
        print(f"❌ خطأ في حفظ النتائج: {e}")

def load_results(filename="results.json"):
    """
    Load results from JSON file
    
    Args:
        filename: File name
    
    Returns:
        dict: Loaded results
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"✅ تم تحميل النتائج من {filename}")
        return results
        
    except Exception as e:
        print(f"❌ خطأ في تحميل النتائج: {e}")
        return {}

def get_feature_importance(results, method="Genetic Algorithm"):
    """
    Get feature importance from specific results
    
    Args:
        results: Results
        method: Method name
    
    Returns:
        numpy.array: Feature importance
    """
    if method in results and 'selected_features' in results[method]:
        return results[method]['selected_features']
    else:
        return None

def compare_methods_performance(results):
    """
    Compare performance of different methods
    
    Args:
        results: Results
    
    Returns:
        dict: Performance comparison
    """
    comparison = {}
    
    for method, metrics in results.items():
        if 'accuracy' in metrics and 'f1' in metrics:
            comparison[method] = {
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'n_features': metrics.get('n_features', 0)
            }
    
    return comparison

def get_best_method(results, metric='accuracy'):
    """
    Get best method based on a specific metric
    
    Args:
        results: Results
        metric: Metric (accuracy, f1)
    
    Returns:
        tuple: (method_name, value)
    """
    best_method = None
    best_value = 0.0
    
    for method, metrics in results.items():
        if metric in metrics:
            value = metrics[metric]
            if value > best_value:
                best_value = value
                best_method = method
    
    return best_method, best_value

def generate_summary_report(results):
    """
    Generate a summary report of results
    
    Args:
        results: Results
    
    Returns:
        str: Summary report
    """
    report = []
    report.append("📊 Summary Report of Comparative Analysis")
    report.append("=" * 50)
    
    # Best method
    best_acc_method, best_acc_value = get_best_method(results, 'accuracy')
    best_f1_method, best_f1_value = get_best_method(results, 'f1')
    
    report.append(f"\n🏆 Best Method (Accuracy): {best_acc_method}")
    report.append(f"   Accuracy: {best_acc_value:.4f}")
    
    report.append(f"\n🏆 Best Method (F1): {best_f1_method}")
    report.append(f"   F1 Score: {best_f1_value:.4f}")
    
    # General comparison
    report.append(f"\n📈 General Comparison:")
    for method, metrics in results.items():
        if 'accuracy' in metrics and 'f1' in metrics:
            report.append(f"   {method}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    return "\n".join(report)
