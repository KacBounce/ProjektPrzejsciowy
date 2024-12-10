import numpy as np
from PIL import Image
import random
import cv2
from threading import Thread, Lock
import time
import matplotlib.pyplot as plt
import numpy as np

delimiter_str = '##END'
delimiter = np.unpackbits(
    np.array([ord(char) for char in delimiter_str], dtype=np.uint8))
image = cv2.imread('Lenna(test_image).png', 0)
genetic_lock = Lock()
pso_lock = Lock()

best_particle = []
best_solution = []


def create_chromosome_array():
    # Creates an array of 0s and 1s.
   return np.random.choice([0, 1], size=(21,))


def hide_bit(value, bit):
    return (value & ~1) | bit  # Replace the least significant bit.W

def Encode(host, binary_message, binary_chromosome):
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
                transformed[h, w] = hide_bit(transformed[h, w], binary_message[message_idx])
                message_idx += 1
            elif delimiter_idx < len(delimiter):
                transformed[h, w] = hide_bit(transformed[h, w], delimiter[delimiter_idx])
                delimiter_idx += 1
            else:
                return transformed

    return transformed

def Decode(host, binary_chromosome):
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

def get_traversal(start_h, start_w, direction, height, width):
# Convert direction to an integer to simplify comparisons
    direction_code = direction[0] * 4 + direction[1] * 2 + direction[2]  # e.g., [1,0,0] becomes 4, [0,0,0] becomes 0
    
    if direction_code == 0:  # Down-Right
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

def image_to_binary_data(image):
    return np.unpackbits(image.astype(np.uint8))

def hide_bit(pixel, bit):
    return (pixel & ~1) | bit

# def fitness_function(binary_chromosome):
#     global image, secret_image, correct_counter, delimiter
#     message = "A3FZ9YBQ" * 100  # 2 MB of 'A'

#     #binary_data = ''.join(format(ord(c), '08b') for c in message)
#     binary_data = ''.join(random.choice(['0', '1']) for _ in range(131072))
#     secret = Encode(image, binary_data, binary_chromosome)
#     decoded_data = Decode(secret, binary_chromosome)
#     original_binary_data = image_to_binary_data(image)
#     secret_binary_data = image_to_binary_data(secret)
#     if (binary_data == decoded_data):correct_counter+=1
#     psnr = cv2.PSNR(image, secret)
#     ber = calculate_BER(original_binary_data, secret_binary_data)
#     return psnr, ber

binary_data = np.random.randint(0, 2, 87500, dtype=np.uint8)
def fitness_function_array(chromosome):
    global image, secret_image, delimiter, correct_counter, binary_data

    # Generate random binary message.
    # 131072 bits (16KB).
    

    # Encode the secret into the image.
    secret = Encode(image, binary_data, chromosome)

    # decoded_data = Decode(secret, chromosome)
    # if np.array_equal(binary_data, decoded_data):
    #     print("CORRECT")
    original_binary_data = image_to_binary_data(image)
    secret_binary_data = image_to_binary_data(secret)
    psnr = cv2.PSNR(image, secret)
    ber = calculate_BER(original_binary_data, secret_binary_data)
    #ber = 0
    return psnr, ber


# def image_to_binary_data(image):
#     pixels = np.array(image).flatten()
#     binary_data = ''.join(format(pixel, '08b') for pixel in pixels)
#     return binary_data

def genetic_with_tracking_array():
    global best_solution
    population_size = 50
    num_generations = 100
    mutation_rate = 0.3
    num_bits = 21
    population = [create_chromosome_array() for _ in range(population_size)]
    psnr_values = []
    ber_values = []

    # Initial best solution
    best_individual = population[0]

    for generation in range(num_generations):
        # Calculate fitness scores and store them for tracking
        fitness_scores = [fitness_function_array(chromosome)[0] for chromosome in population]
        psnr_values.append(max(fitness_scores))  # Track PSNR of the best individual
        best_individual = population[np.argmax(fitness_scores)]
        ber_values.append(fitness_function_array(best_individual)[1])  # Track BER

        # Selection: Select parents based on fitness scores
        selected_parents = random.choices(population, weights=fitness_scores, k=population_size)

        # Crossover: Generate offspring using crossover
        offspring = []
        for i in range(0, population_size, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            crossover_point = np.random.randint(1, num_bits - 1)
            offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            offspring.extend([offspring1, offspring2])

        # Mutation: Apply mutation to the offspring
        for i in range(population_size):
            for j in range(num_bits):
                if np.random.random() < mutation_rate:
                    offspring[i][j] = 1 - offspring[i][j]

        # Elitism: Ensure the best individual is retained at the end of the new population
        best_fitness_index = np.argmax(fitness_scores)
        best_chromosome = population[best_fitness_index]

        # Add the best individual from the previous generation to the new population
        offspring[-1] = best_chromosome
        population = offspring  # Set the new population for the next generation

    best_solution = max(population, key=lambda x: fitness_function_array(x)[0])
    return psnr_values, ber_values






# The main PSO function with improvements
def pso_with_tracking_array():
    num_particles = 50
    num_dimensions = 21  # Number of bits in each binary string (chromosome)
    max_iterations = 100
    c1, c2 = 2.0, 2.0
    global_best_position = None

    # Initialize particles with random binary strings (0s and 1s)
    particles = [create_chromosome_array() for _ in range(num_particles)]
    velocities = [[random.uniform(-1, 1) for _ in range(num_dimensions)]
                  for _ in range(num_particles)]    
    best_positions = particles[:]  # Initialize best positions with initial particles
    best_fitness = [fitness_function_array(p)[0] for p in particles]  # Track fitness for personal bests

    # PSNR and BER tracking for the global best solution
    psnr_values = []
    ber_values = []

    # Initial best solution for elitism
    fitness_scores = [fitness_function_array(p)[0] for p in particles]
    global_best_position = particles[np.argmax(fitness_scores)]

    # PSO main loop
    for iteration in range(max_iterations):
        for i in range(num_particles):
            fitness = fitness_function_array(particles[i])[0]

            # Update the personal best if fitness improves
            if fitness > best_fitness[i]:
                best_fitness[i] = fitness
                best_positions[i] = particles[i].copy()

            # Update the global best if fitness improves
            if global_best_position is None or fitness > fitness_function_array(global_best_position)[0]:
                global_best_position = particles[i].copy()

        # Record PSNR and BER for the global best solution
        psnr_values.append(fitness_function_array(global_best_position)[0])
        ber_values.append(fitness_function_array(global_best_position)[1])

        # Inertia weight for velocity stability
        w = 0.9 - (0.5 * iteration / max_iterations)  # Decaying inertia weight
        
        # Update velocities and positions for each particle
        for i in range(num_particles):
            for j in range(num_dimensions):
                r1, r2 = random.random(), random.random()
                
                # Cognitive and social components of velocity update
                cognitive = c1 * r1 * (int(best_positions[i][j]) - int(particles[i][j]))
                social = c2 * r2 * (int(global_best_position[j]) - int(particles[i][j]))
                
                # Apply inertia weight and update the velocity
                velocities[i][j] = w * velocities[i][j] + cognitive + social

                # Clip velocities to prevent large updates
                velocities[i][j] = np.clip(velocities[i][j], -4, 4)

                # Update position: flip the bit with probability derived from the velocity
                probability = 1 / (1 + np.exp(-velocities[i][j]))  # Sigmoid-based update
                if np.random.random() < probability:
                    particles[i][j] = 1 - particles[i][j]  # Flip the bit

    return psnr_values, ber_values





lock = Lock()

# Function Definitions


def run_genetic():
    global genetic_times, genetic_psnr, genetic_ber
    print("Starting Genetic Algorithm")
    try:
        start_time = time.time()
        g_psnr, g_ber = genetic_with_tracking_array()
        with lock:
            genetic_times.append(time.time() - start_time)
            genetic_psnr.append(g_psnr)
            genetic_ber.append(g_ber)
        print("Finished Genetic Algorithm")
    except Exception as e:
        print(f"Error in Genetic Algorithm: {e}")


def run_pso():
    global pso_times, pso_psnr, pso_ber
    print("Starting PSO")
    try:
        start_time = time.time()
        p_psnr, p_ber = pso_with_tracking_array()
        with lock:
            pso_times.append(time.time() - start_time)
            pso_psnr.append(p_psnr)
            pso_ber.append(p_ber)
        print("Finished PSO")
    except Exception as e:
        print(f"Error in PSO: {e}")


def calculate_BER(original, received):
    # Ensure both sequences are of the same length
    if len(original) != len(received):
        raise ValueError("Both binary sequences must have the same length.")

    # Count bit errors using NumPy
    bit_errors = np.count_nonzero(original != received)

    # Calculate BER
    total_bits = original.size
    ber = bit_errors / total_bits

    # Calculate BER
    total_bits = len(original)
    ber = bit_errors / total_bits

    return ber
# Initialize Data
genetic_times, pso_times = [], []
genetic_psnr, pso_psnr = [], []
genetic_ber, pso_ber = [], []

# Compare Algorithms


def compare_algorithms():
    global pso_times, pso_psnr, pso_ber, genetic_times, genetic_psnr, genetic_ber
    p_threads, g_threads = [], []
    for _ in range(3):  # Run multiple trials for statistical comparison
        print("Starting new iteration")
        genetic_thread = Thread(target=run_genetic)
        pso_thread = Thread(target=run_pso)

        genetic_thread.start()
        pso_thread.start()

        p_threads.append(genetic_thread)
        g_threads.append(pso_thread)
        
        genetic_thread.join()
        pso_thread.join()
        # start_time = time.time()
        # p_psnr, p_ber = pso_with_tracking()
        # pso_times.append(time.time() - start_time)
        # pso_psnr.append(p_psnr)
        # pso_ber.append(p_ber)
        # start_time = time.time()
        # g_psnr, g_ber = genetic_with_tracking()
        # genetic_times.append(time.time() - start_time)
        # genetic_psnr.append(g_psnr)
        # genetic_ber.append(g_ber)
    # for i in p_threads:
    #     i.join()
    #     print("PSO thread completed")
    
    # for i in g_threads:
    #     i.join()
    #     print("Genetic thread completed")

 # Visualization for Genetic Algorithm PSNR and BER
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(genetic_psnr, axis=0), label='Genetic Algorithm PSNR')
    plt.title('Genetic Algorithm PSNR Evolution')
    plt.xlabel('Generations / Iterations')
    plt.ylabel('PSNR')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(genetic_ber, axis=0), label='Genetic Algorithm BER')
    plt.title('Genetic Algorithm BER Evolution')
    plt.xlabel('Generations / Iterations')
    plt.ylabel('BER')
    plt.legend()
    plt.show()

    # Visualization for PSO PSNR and BER
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(pso_psnr, axis=0), label='PSO PSNR')
    plt.title('PSO PSNR Evolution')
    plt.xlabel('Generations / Iterations')
    plt.ylabel('PSNR')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(pso_ber, axis=0), label='PSO BER')
    plt.title('PSO BER Evolution')
    plt.xlabel('Generations / Iterations')
    plt.ylabel('BER')
    plt.legend()
    plt.show()

    # Visualization for Average Execution Time for both Genetic and PSO
    plt.figure(figsize=(10, 6))
    plt.bar(['Genetic', 'PSO'], [np.mean(genetic_times), np.mean(pso_times)])
    plt.title('Average Execution Time')
    plt.ylabel('Time (s)')
    plt.show()


compare_algorithms()