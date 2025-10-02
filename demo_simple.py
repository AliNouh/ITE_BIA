#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified version of the feature selection project using genetic algorithm
This version works without external libraries for demonstration purposes
"""

import random
import math

class SimpleGeneticFeatureSelector:
    """
    Simple implementation of genetic algorithm for feature selection
    Uses simplified fitness function instead of scikit-learn
    """

    def __init__(self, population_size=20, n_generations=15,
                 crossover_rate=0.8, mutation_rate=0.02):
        self.pop_size = population_size
        self.n_gen = n_generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.best_chromosome = None
        self.best_fitness = -float('inf')

    def _init_population(self, n_features):
        """Create initial population of chromosomes"""
        population = []
        for _ in range(self.pop_size):
            # Each chromosome is a list of 0s and 1s representing selected features
            chromosome = [random.randint(0, 1) for _ in range(n_features)]
            # Ensure at least one feature is selected
            if sum(chromosome) == 0:
                chromosome[random.randint(0, n_features-1)] = 1
            population.append(chromosome)
        return population
    
    def _fitness(self, chromosome, X, y):
        """
        Simplified fitness function - in reality would use machine learning model
        Here we use an approximation function for demonstration purposes
        """
        selected_features = [i for i, gene in enumerate(chromosome) if gene == 1]
        if len(selected_features) == 0:
            return 0

        # Simulate model accuracy based on number of selected features
        # In reality, you would train a model and calculate actual accuracy
        n_selected = len(selected_features)
        n_total = len(chromosome)

        # Fitness function prefers reasonable number of features (not too few, not too many)
        optimal_ratio = 0.3  # 30% of features
        ratio = n_selected / n_total

        # Calculate fitness based on proximity to optimal ratio
        fitness = 1.0 - abs(ratio - optimal_ratio)

        # Add small randomness to simulate performance variation
        fitness += random.uniform(-0.1, 0.1)

        return max(0, fitness)

    def _selection(self, population, fitness_scores):
        """Parent selection using Tournament Selection"""
        selected = []
        for _ in range(len(population)):
            # Random selection of three individuals
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)),
                                             min(tournament_size, len(population)))
            # Select the best from the group
            best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[best_idx][:])  # Copy of chromosome
        return selected
    
    def _crossover(self, parent1, parent2):
        """Crossover between two parents using Single Point Crossover"""
        if random.random() > self.cx_rate:
            return parent1[:], parent2[:]

        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def _mutation(self, chromosome):
        """Mutation in chromosome"""
        mutated = chromosome[:]
        for i in range(len(mutated)):
            if random.random() < self.mut_rate:
                mutated[i] = 1 - mutated[i]  # Bit flip

        # Ensure at least one feature is selected
        if sum(mutated) == 0:
            mutated[random.randint(0, len(mutated)-1)] = 1

        return mutated
    
    def fit(self, X, y, verbose=True):
        """Train the genetic algorithm"""
        n_features = len(X[0]) if X else 10  # Default 10 features for demo

        # Create initial population
        population = self._init_population(n_features)

        if verbose:
            print(f"Starting genetic algorithm with {self.pop_size} individuals and {n_features} features")
            print(f"Number of generations: {self.n_gen}")
            print("-" * 50)

        for generation in range(self.n_gen):
            # Calculate fitness for each individual
            fitness_scores = [self._fitness(chrom, X, y) for chrom in population]

            # Track best individual
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_chromosome = population[best_idx][:]

            if verbose and generation % 5 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                n_selected = sum(self.best_chromosome)
                print(f"Generation {generation:2d}: Best fitness = {self.best_fitness:.3f}, "
                      f"Average = {avg_fitness:.3f}, Selected features = {n_selected}")

            # Parent selection
            selected = self._selection(population, fitness_scores)

            # Create new generation
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < len(selected) else selected[0]

                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.pop_size]

        if verbose:
            print("-" * 50)
            print(f"Algorithm completed!")
            print(f"Best fitness: {self.best_fitness:.3f}")
            print(f"Number of selected features: {sum(self.best_chromosome)}")
            print(f"Selected features: {[i for i, gene in enumerate(self.best_chromosome) if gene == 1]}")

        return self
    
    def get_selected_features(self):
        """Return indices of selected features"""
        if self.best_chromosome is None:
            return []
        return [i for i, gene in enumerate(self.best_chromosome) if gene == 1]


def demo_run():
    """Demo run of the algorithm"""
    print("=" * 60)
    print("Feature Selection Project using Genetic Algorithm")
    print("BIA601 - RAFD")
    print("=" * 60)
    print()

    # Demo data (in reality would come from CSV file)
    # Simulate 100 samples with 20 features
    n_samples, n_features = 100, 20
    X = [[random.uniform(-1, 1) for _ in range(n_features)] for _ in range(n_samples)]
    y = [random.randint(0, 1) for _ in range(n_samples)]

    print(f"Demo data: {n_samples} samples, {n_features} features")
    print()

    # Create and train genetic algorithm
    ga = SimpleGeneticFeatureSelector(
        population_size=20,
        n_generations=15,
        crossover_rate=0.8,
        mutation_rate=0.02
    )

    ga.fit(X, y, verbose=True)

    print()
    print("=" * 60)
    print("Final Results:")
    selected_features = ga.get_selected_features()
    print(f"Selected {len(selected_features)} features out of {n_features}")
    print(f"Selected features: {selected_features}")
    print(f"Selection ratio: {len(selected_features)/n_features*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    demo_run()
