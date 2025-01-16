import csv
import csv 
import numpy as np
from PIL import Image
import random
import cv2
from threading import Thread, Lock
import time
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, Process, freeze_support, Manager
import math
import cProfile

best_particle = []
best_solution = []


def create_chromosome_array():
    return np.random.choice([0, 1], size=(21,))


def hide_bit(value, bit):   
    result = np.uint8((value & 0xFE) | bit)
    
    return result

def get_traversal(start_h, start_w, direction, height, width):
    direction_code = direction[0] * 4 + direction[1] * 2 + direction[2]
    if direction_code == 0:  
        return range(start_h, height), range(start_w, width)
    elif direction_code == 1:  # Right-Down
        return range(start_w, width), range(start_h, height)
    elif direction_code == 2:  # Down-Left
        return range(start_h, height), range(width - 1 - start_w, -1, -1)
    elif direction_code == 3:  # Left-Down
        return range(width - 1 - start_w, -1, -1), range(start_h, height)
    elif direction_code == 4:  # Left-Up
        return range(width - 1 - start_w, -1, -1), range(height - 1 - start_h, -1, -1)
    elif direction_code == 5:  # Up-Left
        return range(height - 1 - start_h, -1, -1), range(width - 1 - start_w, -1, -1)
    elif direction_code == 6:  # Up-Right
        return range(height - 1 - start_h, -1, -1), range(start_w, width)
    elif direction_code == 7:  # Right-Up
        return range(start_w, width), range(height - 1 - start_h, -1, -1)

def Encode(host, binary_message, binary_chromosome, delimiter):
    height, width = host.shape
    transformed = host.copy()
    start_h = int(''.join(map(str, binary_chromosome[2:10])), 2)
    start_w = int(''.join(map(str, binary_chromosome[10:18])), 2)
    direction = binary_chromosome[18:21]
    rows, cols = get_traversal(start_h, start_w, direction, height, width)
    message_idx, delimiter_idx = 0, 0
    for h in rows:
        for w in cols:
            if message_idx < len(binary_message):
                transformed[h, w] = hide_bit(
                    transformed[h, w], binary_message[message_idx])
                message_idx += 1
            elif delimiter_idx < len(delimiter):
                transformed[h, w] = hide_bit(
                    transformed[h, w], delimiter[delimiter_idx])
                delimiter_idx += 1
            else:
                return transformed
    return transformed


def Decode(host, binary_chromosome, delimiter):
    height, width = host.shape
    start_h = int(''.join(map(str, binary_chromosome[2:10])), 2)
    start_w = int(''.join(map(str, binary_chromosome[10:18])), 2)
    direction = binary_chromosome[18:21]
    rows, cols = get_traversal(start_h, start_w, direction, height, width)
    max_bits = height * width * 8
    binary_data = np.zeros(max_bits, dtype=np.uint8)
    idx = 0
    for h in rows:
        for w in cols:
            if np.array_equal(binary_data[idx - len(delimiter):idx], delimiter):
                return binary_data[:idx - len(delimiter)]
            bin_pixel_str = format(host[h, w], '08b')
            binary_data[idx] = int(bin_pixel_str[-1])
            idx += 1
    return binary_data[:idx - len(delimiter)] if np.array_equal(binary_data[idx - len(delimiter):idx], delimiter) else binary_data[:idx]


def calculate_BER(original, received):
    if len(original) != len(received):
        raise ValueError("Both binary sequences must have the same length.")
    bit_errors = np.count_nonzero(original != received)
    return bit_errors / original.size


def fitness_function_array(args):
    #print(f"args: {args}") 
    chromosome, image, binary_data, delimiter = args
    secret = Encode(image, binary_data, chromosome, delimiter)
    original_binary_data = np.unpackbits(image.astype(np.uint8))
    secret_binary_data = np.unpackbits(secret.astype(np.uint8))
    psnr = cv2.PSNR(image, secret)
    ber = calculate_BER(original_binary_data, secret_binary_data)
    #print("COmplete")
    return psnr, ber


def parallel_fitness(population, image, binary_data, delimiter, num_processes):
    with Pool(processes=num_processes) as pool:
        fitness_scores = pool.map(fitness_function_array,
                                      [(chromosome, image, binary_data, delimiter) for chromosome in population])
    return fitness_scores


def genetic_with_tracking_array(image, binary_data, delimiter, num_processes):
    population_size = 1000
    num_generations = 100
    mutation_rate = 0.3
    num_bits = 21
    population = [create_chromosome_array() for _ in range(population_size)]

    psnr_values = []
    ber_values = []

    global_best_position = population[0]
    global_best_score = float('-inf')

    for generation in range(num_generations):
        fitness_scores = parallel_fitness(
            population, image, binary_data, delimiter, num_processes)
        psnr_scores = [score[0] for score in fitness_scores]
        ber_scores = [score[1] for score in fitness_scores]

        max_psnr_index = np.argmax(psnr_scores)
        max_psnr_value = psnr_scores[max_psnr_index]
        best_chromosome = population[max_psnr_index]

        if max_psnr_value > global_best_score:
            global_best_score = max_psnr_value
            global_best_position = best_chromosome
            
        psnr_values.append(global_best_score)

        ber_values.append(ber_scores[max_psnr_index])

        selected_parents = random.choices(
            population, weights=psnr_scores, k=population_size)

        offspring = []
        for i in range(0, population_size, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            crossover_point = np.random.randint(1, num_bits - 1)
            offspring1 = np.concatenate(
                [parent1[:crossover_point], parent2[crossover_point:]])
            offspring2 = np.concatenate(
                [parent2[:crossover_point], parent1[crossover_point:]])
            offspring.extend([offspring1, offspring2])

        for i in range(population_size):
            for j in range(num_bits):
                if np.random.random() < mutation_rate:
                    offspring[i][j] = 1 - offspring[i][j]

        offspring[-1] = global_best_position  
        population = offspring 

    return psnr_values, ber_values


def pso_with_tracking_array(image, binary_data, delimiter, num_processes):
    num_particles = 1000
    num_dimensions = 21 
    max_iterations = 100
    c1, c2 = 2.0, 2.0  # Cognitive and social coefficients
    w_max, w_min = 0.9, 0.4  # Inertia weight range

    particles = [create_chromosome_array() for _ in range(num_particles)]
    velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))

    personal_best_positions = particles.copy()
    personal_best_scores = [float('-inf')] * num_particles

    global_best_position = particles[0]
    global_best_score = float('-inf')

    psnr_values = []
    ber_values = []

    for iteration in range(max_iterations):
        fitness_scores = parallel_fitness(
            particles, image, binary_data, delimiter, num_processes)
        psnr_scores = [score[0] for score in fitness_scores]
        ber_scores = [score[1] for score in fitness_scores]

        for i, score in enumerate(psnr_scores):
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i].copy()
            if score > global_best_score:
                global_best_score = score
                global_best_position = particles[i].copy()

        psnr_values.append(global_best_score)
        ber_values.append(ber_scores[np.argmax(psnr_scores)])

        w = w_max - (w_max - w_min) * (iteration / max_iterations)

        for i in range(num_particles):
            for j in range(num_dimensions):
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (personal_best_positions[i][j] - particles[i][j])
                social = c2 * r2 * (global_best_position[j] - particles[i][j])
                velocities[i][j] = w * velocities[i][j] + cognitive + social

                velocities[i][j] = np.clip(velocities[i][j], -4, 4)

                probability = 1 / (1 + np.exp(-velocities[i][j]))
                if random.random() < probability:
                    particles[i][j] = 1 - particles[i][j] 

    return psnr_values, ber_values


def run_genetic(genetic_results, image, binary_data, delimiter):
    print("Starting genetic")
    start_time = time.time()  
    psnr_scores, ber_scores = genetic_with_tracking_array_seq(
        image, binary_data, delimiter)
    end_time = time.time() 
    genetic_results["psnr"] = psnr_scores
    genetic_results["ber"] = ber_scores
    genetic_results["time"] = end_time - start_time  
    print("Genetic finished")


def run_pso(pso_results, image, binary_data, delimiter):
    print("Starting PSO")
    start_time = time.time()  
    psnr_scores, ber_scores = pso_with_tracking_array_seq(
        image, binary_data, delimiter)
    end_time = time.time()  
    pso_results["psnr"] = psnr_scores
    pso_results["ber"] = ber_scores
    pso_results["time"] = end_time - start_time
    print("PSO finished")
    
def run_simulated(simulated_results, image, binary_data, delimiter):
    print("Starting simulated")
    start_time = time.time()  
    psnr_scores, ber_scores = simulated_annealing(
        image, binary_data, delimiter)
    end_time = time.time() 
    simulated_results["psnr"] = psnr_scores
    simulated_results["ber"] = ber_scores
    simulated_results["time"] = end_time - start_time  
    print("Simulated finished")

def run_aco(aco_results, image, binary_data, delimiter):
    print("Starting ACO")
    start_time = time.time()  
    psnr_scores, ber_scores = ant_colony_optimization(
        image, binary_data, delimiter)
    end_time = time.time() 
    aco_results["psnr"] = psnr_scores
    aco_results["ber"] = ber_scores
    aco_results["time"] = end_time - start_time  
    print("ACO finished")


def genetic_with_tracking_array_seq(image, binary_data, delimiter):
    global best_solution
    population_size = 100
    num_generations = 100
    mutation_rate = 0.3
    num_bits = 21
    population = [create_chromosome_array() for _ in range(population_size)]
    psnr_values = []
    ber_values = []

    # Evaluate initial population
    fitness_results = [fitness_function_array((chromosome, image, binary_data, delimiter))
                       for chromosome in population]
    fitness_scores = [result[0] for result in fitness_results]
    ber_scores = [result[1] for result in fitness_results]

    # Track the best individual from the initial population
    best_fitness_index = np.argmax(fitness_scores)
    best_individual = population[best_fitness_index]
    best_fitness = fitness_scores[best_fitness_index]
    best_ber = ber_scores[best_fitness_index]

    psnr_values.append(best_fitness)
    ber_values.append(best_ber)

    for generation in range(num_generations):
        # Select parents based on fitness scores
        selected_parents = random.choices(
            population, weights=fitness_scores, k=population_size)

        # Create offspring through crossover
        offspring = []
        for i in range(0, population_size, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            crossover_point = np.random.randint(1, num_bits - 1)
            offspring1 = np.concatenate(
                [parent1[:crossover_point], parent2[crossover_point:]])
            offspring2 = np.concatenate(
                [parent2[:crossover_point], parent1[crossover_point:]])
            offspring.extend([offspring1, offspring2])

        # Apply mutation
        for i in range(population_size):
            for j in range(num_bits):
                if np.random.random() < mutation_rate:
                    offspring[i][j] = 1 - offspring[i][j]

        # Replace population with offspring
        population = offspring

        # Evaluate the new population
        fitness_results = [fitness_function_array((chromosome, image, binary_data, delimiter))
                           for chromosome in population]
        fitness_scores = [result[0] for result in fitness_results]
        ber_scores = [result[1] for result in fitness_results]

        # Update the best individual
        best_fitness_index = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[best_fitness_index]
        current_best_ber = ber_scores[best_fitness_index]
        current_best_individual = population[best_fitness_index]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_ber = current_best_ber
            best_individual = current_best_individual

        # Track global best PSNR and BER
        psnr_values.append(best_fitness)
        ber_values.append(best_ber)

        # Elitism: Retain the best individual in the next generation
        population[-1] = best_individual

    # Final best solution
    best_solution = best_individual
    return psnr_values, ber_values



def pso_with_tracking_array_seq(image, binary_data, delimiter):
    num_particles = 100
    num_dimensions = 21
    max_iterations = 100
    c1, c2 = 2.0, 2.0
    global_best_position = None

    particles = [create_chromosome_array() for _ in range(num_particles)]
    velocities = [[random.uniform(-1, 1) for _ in range(num_dimensions)]
                  for _ in range(num_particles)]
    best_positions = particles[:]
    fitness_results = [fitness_function_array((p, image, binary_data, delimiter)) for p in particles]
    best_fitness = [result[0] for result in fitness_results]
    best_ber = [result[1] for result in fitness_results]

    global_best_position = particles[np.argmax(best_fitness)]
    global_best_fitness = max(best_fitness)
    global_best_ber = best_ber[np.argmax(best_fitness)]

    psnr_values = []
    ber_values = []

    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Evaluate the particle
            psnr, ber = fitness_function_array((particles[i], image, binary_data, delimiter))

            # Update the particle's personal best
            if psnr > best_fitness[i]:
                best_fitness[i] = psnr
                best_positions[i] = particles[i].copy()
                best_ber[i] = ber

            # Update the global best position
            if psnr > global_best_fitness:
                global_best_position = particles[i].copy()
                global_best_fitness = psnr
                global_best_ber = ber

        # Store PSNR and BER values of the global best
        psnr_values.append(global_best_fitness)
        ber_values.append(global_best_ber)

        # Decrease inertia weight over iterations
        w = 0.9 - (0.5 * iteration / max_iterations)

        # Update particle velocities and positions
        for i in range(num_particles):
            for j in range(num_dimensions):
                r1, r2 = random.random(), random.random()

                cognitive = c1 * r1 * \
                    (int(best_positions[i][j]) - int(particles[i][j]))
                social = c2 * r2 * \
                    (int(global_best_position[j]) - int(particles[i][j]))

                velocities[i][j] = w * velocities[i][j] + cognitive + social

                # Clip velocities to prevent excessive updates
                velocities[i][j] = np.clip(velocities[i][j], -4, 4)

                # Sigmoid-based update
                probability = 1 / (1 + np.exp(-velocities[i][j]))
                if np.random.random() < probability:
                    particles[i][j] = 1 - particles[i][j]  # Flip the bit

    return psnr_values, ber_values


def ant_colony_optimization(image, binary_data, delimiter):
    num_ants = 100
    max_iterations = 100
    alpha = 1.0  # Pheromone influence
    beta = 2.0   # Heuristic influence
    evaporation_rate = 0.5  # Pheromone evaporation rate
    pheromones = np.ones((num_ants, 21))  # Pheromone matrix

    best_solution = None
    best_psnr = float('-inf')
    best_ber = None

    psnr_values = []
    ber_values = []

    for iteration in range(max_iterations):
        solutions = []
        psnr_scores = []
        ber_scores = []
        
        # Ants explore the solution space
        for ant in range(num_ants):
            solution = create_chromosome_array()
            for i in range(len(solution)):
                # Use pheromone probability to decide the next bit
                pheromone_prob = pheromones[ant][i] ** alpha
                heuristic_prob = (1 / (1 + np.exp(-solution[i]))) ** beta
                transition_prob = pheromone_prob * heuristic_prob
                if np.random.random() < transition_prob:
                    solution[i] = 1 - solution[i]
            solutions.append(solution)

            # Evaluate the solution: single call to fitness function
            psnr, ber = fitness_function_array((solution, image, binary_data, delimiter))
            psnr_scores.append(psnr)
            ber_scores.append(ber)

            # Update the best solution found
            if psnr > best_psnr:
                best_solution = solution
                best_psnr = psnr
                best_ber = ber

        # Pheromone update
        for ant in range(num_ants):
            for i in range(len(pheromones[ant])):
                pheromones[ant][i] = (1 - evaporation_rate) * pheromones[ant][i]
                if psnr_scores[ant] == best_psnr:  # Add pheromone for the best solution
                    pheromones[ant][i] += 1.0

        # Elitism: Ensure the best solution found so far is retained
        for i in range(len(pheromones[0])):
            # Reinforce pheromones for the best solution
            pheromones[0][i] = (1 - evaporation_rate) * pheromones[0][i]
            if best_solution is not None and best_solution[i] == 1:
                pheromones[0][i] += 1.0

        # Store PSNR and BER values
        psnr_values.append(best_psnr)
        ber_values.append(best_ber)

    return psnr_values, ber_values



def simulated_annealing(image, binary_data, delimiter):
    current_solution = create_chromosome_array()
    current_psnr, current_ber = fitness_function_array((current_solution, image, binary_data, delimiter))
    temperature = 1000000.0
    cooling_rate = 0.9999
    num_iterations = 100

    best_solution = current_solution
    best_psnr = current_psnr
    best_ber = current_ber

    psnr_values = []
    ber_values = []

    for iteration in range(num_iterations):
        # Create a neighbor by flipping a random bit
        neighbor = current_solution.copy()
        flip_index = np.random.randint(len(current_solution))
        neighbor[flip_index] = 1 - neighbor[flip_index]
        
        # Evaluate the neighbor
        neighbor_psnr, neighbor_ber = fitness_function_array((neighbor, image, binary_data, delimiter))
        
        # Calculate acceptance probability
        delta_psnr = neighbor_psnr - current_psnr
        if delta_psnr > 0 or np.random.rand() < math.exp(delta_psnr / temperature):
            current_solution = neighbor
            current_psnr = neighbor_psnr
            current_ber = neighbor_ber
        
        # Update best solution using elitism
        if current_psnr > best_psnr:
            best_solution = current_solution
            best_psnr = current_psnr
            best_ber = current_ber

        # Ensure the best solution is retained (elitism)
        if best_psnr > current_psnr:
            current_solution = best_solution
            current_psnr = best_psnr
            current_ber = best_ber

        # Store the PSNR and BER values
        psnr_values.append(best_psnr)
        ber_values.append(best_ber)

        # Cool down the temperature
        temperature *= cooling_rate

    return psnr_values, ber_values


def compare_algorithms(image, binary_data, delimiter):
    manager = Manager()
    genetic_results = manager.dict()
    pso_results = manager.dict()
    simulated_results = manager.dict()
    aco_results = manager.dict()

    genetic_process = Process(target=run_genetic, args=(
        genetic_results, image, binary_data, delimiter))
    pso_process = Process(target=run_pso, args=(
        pso_results, image, binary_data, delimiter))
    aco_process = Process(target=run_aco, args=(
        aco_results, image, binary_data, delimiter))
    simulated_process = Process(target=run_simulated, args=(
        simulated_results, image, binary_data, delimiter))

    # genetic_process.start()
    # pso_process.start()
    # aco_process.start()
    simulated_process.start()

    # genetic_process.join()
    # pso_process.join()
    # aco_process.join()
    simulated_process.join()

    print(f"Genetic PSNR: {genetic_results.get('psnr', 'Not Found')}")
    print(f"PSO PSNR: {pso_results.get('psnr', 'Not Found')}")
    print(f"ACO PSNR: {aco_results.get('psnr', 'Not Found')}")
    print(f"Simulated PSNR: {simulated_results.get('psnr', 'Not Found')}")
    print(f"Genetic Time: {genetic_results.get('time', 'Not Found')} seconds")
    print(f"PSO Time: {pso_results.get('time', 'Not Found')} seconds")
    print(f"ACO Time: {aco_results.get('time', 'Not Found')}")
    print(f"Simulated Time: {simulated_results.get('time', 'Not Found')}")

    # Plot combined PSNR, BER, and time
    plt.figure(figsize=(18, 12))

    # Plot PSNR comparison
    plt.subplot(3, 1, 1)
    # plt.plot(genetic_results["psnr"], label="Genetic PSNR")
    # plt.plot(pso_results["psnr"], label="PSO PSNR")
    # plt.plot(aco_results["psnr"], label="ACO PSNR")
    plt.plot(simulated_results["psnr"], label="Simulated PSNR")
    plt.title('PSNR Comparison')
    plt.xlabel('Generation')
    plt.ylabel('PSNR')
    plt.legend()

    # Plot BER comparison
    plt.subplot(3, 1, 2)
    # plt.plot(genetic_results["ber"], label="Genetic BER")
    # plt.plot(pso_results["ber"], label="PSO BER")
    # plt.plot(aco_results["ber"], label="ACO BER")
    plt.plot(simulated_results["ber"], label="Simulated BER")
    plt.title('BER Comparison')
    plt.xlabel('Generation')
    plt.ylabel('BER')
    plt.legend()

    # # Plot time comparison
    # plt.subplot(3, 1, 3)
    # times = [
    #     genetic_results.get('time', 0),
    #     pso_results.get('time', 0),
    #     aco_results.get('time', 0),
    #     simulated_results.get('time', 0)
    # ]
    # algorithms = ['Genetic', 'PSO', 'ACO', 'Simulated']
    # plt.bar(algorithms, times, color=['blue', 'orange', 'green', 'red'])
    # plt.title('Time Comparison')
    # plt.xlabel('Algorithm')
    # plt.ylabel('Time (seconds)')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # Global Constants
    delimiter_str = '##END'
    delimiter = np.unpackbits(
        np.array([ord(char) for char in delimiter_str], dtype=np.uint8))
    image = cv2.imread('Lenna(test_image).png', 0)
    image = cv2.resize(image, (840,840))
    binary_data = np.random.randint(0, 2, math.ceil(3/5 * 840 * 840), dtype=np.uint8)
    compare_algorithms(image, binary_data, delimiter)


# def run_sequential_for_size(image, delimiter, matrix_size):
#     print(f"Starting sequential run for matrix size {matrix_size}...")

#     binary_data = np.random.randint(
#         0, 2, math.ceil(matrix_size * 3/5), dtype=np.uint8)
#     cropped_image = cv2.resize(
#         image, (matrix_size, matrix_size), interpolation=cv2.INTER_LINEAR)

#     print(f"  - Starting sequential genetic algorithm...")
#     start_time = time.time()
#     psnr_gen_seq, ber_gen_seq = genetic_with_tracking_array_seq(
#         cropped_image, binary_data, delimiter)
#     end_time = time.time()
#     sequential_time_gen = end_time - start_time
#     print(
#         f"  - Sequential genetic algorithm completed in {sequential_time_gen:.4f} seconds.")

#     print(f"  - Starting sequential PSO algorithm...")
#     start_time = time.time()
#     psnr_gen_seq, ber_gen_seq = pso_with_tracking_array_seq(
#         cropped_image, binary_data, delimiter)
#     end_time = time.time()
#     sequential_time_pso = end_time - start_time
#     print(
#         f"  - Sequential PSO algorithm completed in {sequential_time_pso:.4f} seconds.")

#     return sequential_time_gen, sequential_time_pso


# def run_parallel_for_size(image, delimiter, matrix_size, num_processes, sequential_times):
#     print(f"Starting parallel run for matrix size {matrix_size} with {num_processes} processes...")

#     binary_data = np.random.randint(
#         0, 2, math.ceil(matrix_size * 3/5), dtype=np.uint8)
#     cropped_image = cv2.resize(
#         image, (matrix_size, matrix_size), interpolation=cv2.INTER_LINEAR)

#     print(f"  - Starting parallel genetic algorithm with {num_processes} processes...")
#     start_time = time.time()
#     psnr_gen_par, ber_gen_par = genetic_with_tracking_array(
#         cropped_image, binary_data, delimiter, num_processes)
#     end_time = time.time()
#     parallel_time_gen = end_time - start_time
#     print(
#         f"  - Parallel genetic algorithm completed in {parallel_time_gen:.4f} seconds.")

#     print(f"  - Starting parallel PSO algorithm with {num_processes} processes...")
#     start_time = time.time()
#     psnr_gen_par, ber_gen_par = pso_with_tracking_array(
#         cropped_image, binary_data, delimiter, num_processes)
#     end_time = time.time()
#     parallel_time_pso = end_time - start_time
#     print(
#         f"  - Parallel PSO algorithm completed in {parallel_time_pso:.4f} seconds.")

#     return matrix_size, num_processes, sequential_times[0], parallel_time_gen, sequential_times[1], parallel_time_pso


# def compare_algorithms(image, delimiter, matrix_sizes, num_processes):
#     print(f"Starting comparison of algorithms across different image sizes...")
#     sequential_results = {}
#     for matrix_size in matrix_sizes:
#         print(f"Running sequential algorithms for matrix size: {matrix_size}")
#         sequential_times = run_sequential_for_size(
#             image, delimiter, matrix_size)
#         sequential_results[matrix_size] = sequential_times

#     # for num_proc in num_processes:
#     #     csv_filename = f"result_{num_proc}.txt"
#     #     with open(csv_filename, mode='w', newline='') as file:
#     #         writer = csv.writer(file)
#     #         writer.writerow(['MatrixSize', 'NumProcesses', 'SequentialTime_gen', 'ParallelTime_gen', 'SequentialTime_pso', 'ParallelTime_pso'])  # Header

#     #         for matrix_size in matrix_sizes:
#     #             print(f"Running for matrix size: {matrix_size} and num_processes: {num_proc}")
#     #             sequential_time_gen, sequential_time_pso = 0,0
#     #             result = run_parallel_for_size(
#     #                 image, delimiter, matrix_size, num_proc, (sequential_time_gen, sequential_time_pso))
#     #             writer.writerow(result)

#     #     print(f"Results saved to {csv_filename}")

# if __name__ == '__main__':
#     image = cv2.imread('Lenna(test_image)_512.png', 0)
#     delimiter_str = '##END'
#     delimiter = np.unpackbits(
#         np.array([ord(char) for char in delimiter_str], dtype=np.uint8))

#     matrix_sizes = [360, 480, 600, 720, 840]
#     num_processes = [24]
#     compare_algorithms(image, delimiter, matrix_sizes, num_processes)