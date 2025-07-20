import numpy as np
import random
from evolution.genome import create_genome
from environments.cartpole_runner import evaluate_genome

class GeneticAlgorithm:
    def __init__(self, input_size, hidden_size, output_size, population_size, mutation_rate, render):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.genome_length = self.input_size * self.hidden_size + self.hidden_size + self.hidden_size * self.output_size + self.output_size
        self.population = self.initialize_population()
        self.render = render

    def initialize_population(self):
        return [create_genome(self.input_size, self.hidden_size, self.output_size) 
            for _ in range(self.population_size)]

    def mutate(self, genome):
        mutation = np.random.randn(len(genome)) * self.mutation_rate
        return genome + mutation

    def evaluate_population(self, generation):
        fitness_scores = []
        for i, genome in enumerate(self.population):
            fitness = evaluate_genome(genome, self.input_size, self.hidden_size, self.output_size, i, generation, self.render)
            fitness_scores.append((genome, fitness))

        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return fitness_scores

    def select_parents(self, sorted_population):
        num_parents = max(2, int(self.population_size * 0.2))
        parents = [genome for genome, _ in sorted_population[:num_parents]]
        return parents

    def crossover(self, parent1, parent2):
        child = np.array([np.random.choice([g1, g2])
                          for g1, g2 in zip(parent1, parent2)])
        return child

    def next_generation(self, sorted_popilation):
        parents = self.select_parents(sorted_popilation)
        new_population = []

        new_population.append(parents[0])
        
        while len(new_population) < self.population_size:
            parent1, parent2 = random.choices(parents, k=2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
