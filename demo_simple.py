#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نسخة مبسطة من مشروع اختيار الميزات باستخدام الخوارزمية الوراثية
هذه النسخة تعمل بدون مكتبات خارجية لأغراض العرض التوضيحي
"""

import random
import math

class SimpleGeneticFeatureSelector:
    """
    تنفيذ مبسط للخوارزمية الوراثية لاختيار الميزات
    يستخدم دالة fitness مبسطة بدلاً من scikit-learn
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
        """إنشاء مجتمع أولي من الكروموسومات"""
        population = []
        for _ in range(self.pop_size):
            # كل كروموسوم هو قائمة من 0 و 1 تمثل الميزات المختارة
            chromosome = [random.randint(0, 1) for _ in range(n_features)]
            # تأكد من وجود ميزة واحدة على الأقل
            if sum(chromosome) == 0:
                chromosome[random.randint(0, n_features-1)] = 1
            population.append(chromosome)
        return population
    
    def _fitness(self, chromosome, X, y):
        """
        دالة fitness مبسطة - في الواقع ستستخدم نموذج تعلم آلة
        هنا نستخدم دالة تقريبية للعرض التوضيحي
        """
        selected_features = [i for i, gene in enumerate(chromosome) if gene == 1]
        if len(selected_features) == 0:
            return 0
        
        # محاكاة دقة النموذج بناءً على عدد الميزات المختارة
        # في الواقع، ستقوم بتدريب نموذج وحساب الدقة الفعلية
        n_selected = len(selected_features)
        n_total = len(chromosome)
        
        # دالة fitness تفضل عدد معقول من الميزات (لا قليل جداً ولا كثير جداً)
        optimal_ratio = 0.3  # 30% من الميزات
        ratio = n_selected / n_total
        
        # حساب fitness بناءً على قرب النسبة من النسبة المثلى
        fitness = 1.0 - abs(ratio - optimal_ratio)
        
        # إضافة عشوائية صغيرة لمحاكاة تباين الأداء
        fitness += random.uniform(-0.1, 0.1)
        
        return max(0, fitness)
    
    def _selection(self, population, fitness_scores):
        """اختيار الوالدين باستخدام Tournament Selection"""
        selected = []
        for _ in range(len(population)):
            # اختيار عشوائي لثلاثة أفراد
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), 
                                             min(tournament_size, len(population)))
            # اختيار الأفضل من المجموعة
            best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[best_idx][:])  # نسخة من الكروموسوم
        return selected
    
    def _crossover(self, parent1, parent2):
        """تهجين بين والدين باستخدام Single Point Crossover"""
        if random.random() > self.cx_rate:
            return parent1[:], parent2[:]
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutation(self, chromosome):
        """طفرة في الكروموسوم"""
        mutated = chromosome[:]
        for i in range(len(mutated)):
            if random.random() < self.mut_rate:
                mutated[i] = 1 - mutated[i]  # قلب البت
        
        # تأكد من وجود ميزة واحدة على الأقل
        if sum(mutated) == 0:
            mutated[random.randint(0, len(mutated)-1)] = 1
            
        return mutated
    
    def fit(self, X, y, verbose=True):
        """تدريب الخوارزمية الوراثية"""
        n_features = len(X[0]) if X else 10  # افتراضي 10 ميزات للعرض
        
        # إنشاء المجتمع الأولي
        population = self._init_population(n_features)
        
        if verbose:
            print(f"بدء الخوارزمية الوراثية مع {self.pop_size} فرد و {n_features} ميزة")
            print(f"عدد الأجيال: {self.n_gen}")
            print("-" * 50)
        
        for generation in range(self.n_gen):
            # حساب fitness لكل فرد
            fitness_scores = [self._fitness(chrom, X, y) for chrom in population]
            
            # تتبع أفضل فرد
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_chromosome = population[best_idx][:]
            
            if verbose and generation % 5 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                n_selected = sum(self.best_chromosome)
                print(f"الجيل {generation:2d}: أفضل fitness = {self.best_fitness:.3f}, "
                      f"متوسط = {avg_fitness:.3f}, ميزات مختارة = {n_selected}")
            
            # اختيار الوالدين
            selected = self._selection(population, fitness_scores)
            
            # إنشاء الجيل الجديد
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
            print(f"انتهت الخوارزمية!")
            print(f"أفضل fitness: {self.best_fitness:.3f}")
            print(f"عدد الميزات المختارة: {sum(self.best_chromosome)}")
            print(f"الميزات المختارة: {[i for i, gene in enumerate(self.best_chromosome) if gene == 1]}")
        
        return self
    
    def get_selected_features(self):
        """إرجاع فهارس الميزات المختارة"""
        if self.best_chromosome is None:
            return []
        return [i for i, gene in enumerate(self.best_chromosome) if gene == 1]


def demo_run():
    """تشغيل تجريبي للخوارزمية"""
    print("=" * 60)
    print("مشروع اختيار الميزات باستخدام الخوارزمية الوراثية")
    print("BIA601 - RAFD")
    print("=" * 60)
    print()
    
    # بيانات تجريبية (في الواقع ستأتي من ملف CSV)
    # محاكاة 100 عينة مع 20 ميزة
    n_samples, n_features = 100, 20
    X = [[random.uniform(-1, 1) for _ in range(n_features)] for _ in range(n_samples)]
    y = [random.randint(0, 1) for _ in range(n_samples)]
    
    print(f"البيانات التجريبية: {n_samples} عينة، {n_features} ميزة")
    print()
    
    # إنشاء وتدريب الخوارزمية الوراثية
    ga = SimpleGeneticFeatureSelector(
        population_size=20,
        n_generations=15,
        crossover_rate=0.8,
        mutation_rate=0.02
    )
    
    ga.fit(X, y, verbose=True)
    
    print()
    print("=" * 60)
    print("النتائج النهائية:")
    selected_features = ga.get_selected_features()
    print(f"تم اختيار {len(selected_features)} ميزة من أصل {n_features}")
    print(f"الميزات المختارة: {selected_features}")
    print(f"نسبة الاختيار: {len(selected_features)/n_features*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    demo_run()
