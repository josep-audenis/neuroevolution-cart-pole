import matplotlib.pyplot as plt
from evolution.genetic_algorithm import GeneticAlgorithm

input_size = 4
hidden_size = 10
output_size = 1

population_size = 50
mutation_rate = 0.1
generations = 100

ga = GeneticAlgorithm(input_size, hidden_size, output_size, population_size, mutation_rate)
best_fitness_history = []
avg_fitness_history = []
worst_fitness_history = []

for gen in range(generations):
    sorted_population = ga.evaluate_population()
    fitness_values = [fitness for _, fitness in sorted_population]
    best_fitness = fitness_values[0]
    worst_fitness = fitness_values[-1]
    avg_fitness = sum(fitness_values) / len(fitness_values)
    
    print(f"Generation {gen + 1} | Best: {best_fitness} | Avg: {avg_fitness} | Worst: {worst_fitness}")

    best_fitness_history.append(best_fitness)
    avg_fitness_history.append(avg_fitness)
    worst_fitness_history.append(worst_fitness)
    
    ga.next_generation(sorted_population)

plt.plot(best_fitness_history, label="Best Fitness")
plt.plot(avg_fitness_history, label="Average Fitness")
plt.plot(worst_fitness_history, label="Worst Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("CartPole GA: Best vs Average Fitness")
plt.legend()
plt.show()
