import random

def generate_population(size, puzzle):
    population = []
    for _ in range(size):
        individual = puzzle.copy()
        for i in range(9):
            row = list(range(1, 10))
            for j in range(9):
                if individual[i][j] == 0:
                    random.shuffle(row)
                    individual[i][j] = row.pop()
        population.append(individual)
    return population

def fitness(individual):
    conflicts = 0
    for i in range(9):
        row = individual[i]
        col = [individual[j][i] for j in range(9)]
        conflicts += (len(row) - len(set(row))) + (len(col) - len(set(col)))
    
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            square = []
            for x in range(3):
                for y in range(3):
                    square.append(individual[i+x][j+y])
            conflicts += len(square) - len(set(square))
            
    return 1 / (conflicts + 1)

def crossover(parent1, parent2):
    crossover_point = random.randint(0, 8)
    child = []
    for i in range(9):
        if i <= crossover_point:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def mutate(individual):
    mutation_row = random.randint(0, 8)
    mutation_col = random.randint(0, 8)
    mutation_value = random.randint(1, 9)
    individual[mutation_row][mutation_col] = mutation_value
    return individual

def genetic_algorithm(puzzle, population_size=1000, generations=10000):
    population = generate_population(population_size, puzzle)
    
    for generation in range(generations):
        population = sorted(population, key=lambda x: fitness(x), reverse=True)
        if fitness(population[0]) == 1:
            return population[0]
        
        new_population = []
        for _ in range(population_size // 2):
            parent1 = random.choice(population[:population_size//2])
            parent2 = random.choice(population[:population_size//2])
            child = crossover(parent1, parent2)
            if random.random() < 0.1:
                child = mutate(child)
            new_population.append(child)
            
        population = new_population

    return None

# Example Sudoku puzzle (0 denotes empty cells)
sudoku_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

solution = genetic_algorithm(sudoku_puzzle)

if solution is not None:
    for row in solution:
        print(row)
else:
    print("No solution found.")
