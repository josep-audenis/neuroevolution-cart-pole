import matplotlib.pyplot as plt
from evolution.genetic_algorithm import GeneticAlgorithm

input_size = 4
hidden_size = 8
output_size = 1

population_size = 20
mutation_rate = 0.01
generations = 30

ga = GeneticAlgorithm(input_size, hidden_size, output_size, population_size, mutation_rate)
fitness_history = []

for gen in range(generations):
    sorted_population = ga.evaluate_population()
    best_fitness = sorted_population[0][1]
    print(f"Generation {gen + 1} | Best fitness: {best_fitness}")
    fitness_history.append(best_fitness)
    ga.next_generation(sorted_population)

plt.plot(fitness_history)
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.title("CartPole Genetic Algorithm Evolution")
plt.show()
