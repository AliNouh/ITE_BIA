"""
Pure implementation of Genetic Algorithm for feature selection without sklearn dependencies.
"""
import numpy as np
from random import random, randint, sample
from math import sqrt
from itertools import combinations

class GeneticFeatureSelector:
    """Implementation of a pure Genetic Algorithm for feature selection."""
    
    def __init__(self, population_size=30, n_gen=40,
                 crossover_rate=0.8, mutation_rate=0.02, 
                 tournament_size=3, random_state=42):
        self.pop_size = population_size
        self.n_gen = n_gen
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.tournament_size = tournament_size
        np.random.seed(random_state)
        
        self.best_chromosome_ = None
        self.best_score_ = -1
        self.selected_features_ = None
        
    def _calculate_accuracy(self, X_subset, y):
        """Calculate simple accuracy using k-fold cross validation.
        
        Implements a basic k-nearest neighbors approach for classification.
        """
        if X_subset.shape[1] == 0:
            return 0.0
            
        # Simple k-fold split
        k = 5
        fold_size = len(y) // k
        scores = []
        
        for i in range(k):
            # Split data
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < k - 1 else len(y)
            
            X_train = np.concatenate([X_subset[:test_start], X_subset[test_end:]])
            y_train = np.concatenate([y[:test_start], y[test_end:]])
            X_test = X_subset[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Simple majority voting using 3 nearest neighbors
            predictions = []
            for x_t in X_test:
                # Calculate distances to all training samples
                distances = np.sqrt(np.sum((X_train - x_t) ** 2, axis=1))
                # Get indices of 3 nearest neighbors
                nearest = np.argsort(distances)[:3]
                # Predict by majority vote
                pred = np.bincount(y_train[nearest]).argmax()
                predictions.append(pred)
            
            # Calculate accuracy for this fold
            acc = np.mean(np.array(predictions) == y_test)
            scores.append(acc)
        
        return np.mean(scores)
    
    def _init_population(self, n_features):
        """Initialize population with binary chromosomes."""
        pop = []
        for _ in range(self.pop_size):
            # Create random binary chromosome
            chrom = np.random.choice([0,1], size=n_features, p=[0.7,0.3])
            # Ensure at least one feature is selected
            if np.sum(chrom) == 0:
                chrom[randint(0, n_features-1)] = 1
            pop.append(chrom)
        return pop
    
    def _fitness(self, chromosome, X, y):
        """Calculate fitness score for a chromosome."""
        selected = np.where(chromosome == 1)[0]
        if len(selected) == 0:
            return 0.0
        
        X_selected = X[:, selected]
        accuracy = self._calculate_accuracy(X_selected, y)
        
        # Penalize for too many features
        n_features_penalty = 0.001 * len(selected)
        return accuracy - n_features_penalty
    
    def _tournament_selection(self, population, fitnesses):
        """Select parents using tournament selection."""
        selected = []
        for _ in range(len(population)):
            # Select random candidates for tournament
            candidates = sample(range(len(population)), self.tournament_size)
            # Select the best candidate
            winner = max(candidates, key=lambda i: fitnesses[i])
            selected.append(population[winner].copy())
        return selected
    
    def _crossover(self, parent1, parent2):
        """Perform single-point crossover."""
        if random() > self.cx_rate:
            return parent1.copy(), parent2.copy()
            
        point = randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def _mutate(self, chromosome):
        """Apply bit-flip mutation."""
        for i in range(len(chromosome)):
            if random() < self.mut_rate:
                chromosome[i] = 1 - chromosome[i]
        # Ensure at least one feature is selected
        if np.sum(chromosome) == 0:
            chromosome[randint(0, len(chromosome)-1)] = 1
        return chromosome
    
    def fit(self, X, y):
        """Run genetic algorithm to select best features."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty input arrays")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
            
        n_features = X.shape[1]
        population = self._init_population(n_features)
        
        for generation in range(self.n_gen):
            # Calculate fitness for all chromosomes
            fitnesses = [self._fitness(chrom, X, y) for chrom in population]
            
            # Update best solution
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_score_:
                self.best_score_ = fitnesses[best_idx]
                self.best_chromosome_ = population[best_idx].copy()
            
            # Select parents
            selected = self._tournament_selection(population, fitnesses)
            
            # Create next generation
            next_gen = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[i+1] if i+1 < len(selected) else selected[0]
                c1, c2 = self._crossover(p1, p2)
                next_gen.extend([self._mutate(c1), self._mutate(c2)])
            
            population = next_gen[:self.pop_size]
        
        self.selected_features_ = np.where(self.best_chromosome_ == 1)[0]
        return self
    
    def transform(self, X):
        """Transform X by selecting best features."""
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)