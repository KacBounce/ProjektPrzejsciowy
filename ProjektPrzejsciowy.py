import numpy as np
from PIL import Image
import random
import cv2
from threading import Thread, Lock
import time
import matplotlib.pyplot as plt
import numpy as np

delimiter = '##END'
delimiter = ''.join(format(ord(char), '08b') for char in delimiter)
image = cv2.imread('Lenna(test_image).png', 0)
genetic_lock = Lock()
pso_lock = Lock()

best_particle = ''
best_solution = ''


secret_image = Image.open('Secret Lenna.png', 'r')
secret_width, secret_height = secret_image.size
secret_image = secret_image.convert('L')


def create_chromosome():
    chromosome = random.randint(0, 2097152)
    binary_chromosome = format(chromosome, 'b')
    while (len(binary_chromosome) < 21):
        binary_chromosome = "0" + binary_chromosome
    return binary_chromosome


def transform_secret_image(secret_image, binary_chromosome):
    binary_data = ''.join(format(pixel, '08b')
                          for pixel in secret_image.tobytes())
    if (binary_chromosome[1] == '1'):
        binary_data = ''.join(
            '1' if bit == '0' else '0' for bit in binary_data)
    if (binary_chromosome[0] == '1'):
        binary_data = binary_data[::-1]
    return binary_data


def transform_bits_image(binary_data, width, height, binary_chromosome):
    mode = 'L'  # Grayscale mode

    # Create an empty image
    secret_image = Image.new(mode, (width, height))

    if (binary_chromosome[1] == '1'):
        binary_data = ''.join(
            '1' if bit == '0' else '0' for bit in binary_data)
    if (binary_chromosome[0] == '1'):
        binary_data = binary_data[::-1]

    # Parse the list of bits and set pixel values
    pixels = [int(binary_data[i:i+8], 2)
              for i in range(0, len(binary_data), 8)]
    secret_image.putdata(pixels)
    return secret_image


def hide_bit(value, bit):
    bin_pixel_str = format(value, 'b')
    bin_pixel = list(bin_pixel_str)
    if (bin_pixel[len(bin_pixel) - 1] != bit):
        bin_pixel[len(bin_pixel) - 1] = bit
    bin_pixel_str = "".join(bin_pixel)
    return int(bin_pixel_str, 2)


def Encode(host, binary_message, binary_chromosome):
    global delimiter
    height, width = host.shape
    transformed = host.copy()
    index = 0
    index2 = 0
    # first direction
    if (binary_chromosome[18:21] == "000"):
        for h in range(int(binary_chromosome[2:10], 2), height):
            for w in range(int(binary_chromosome[10:18], 2), width):
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(
                        transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if (index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(
                            transformed[h][w], delimiter[index2])
                        index2 += 1
    # second direction
    elif (binary_chromosome[18:21] == "001"):
        for w in range(int(binary_chromosome[10:18], 2), width):
            for h in range(int(binary_chromosome[2:10], 2), height):
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(
                        transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if (index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(
                            transformed[h][w], delimiter[index2])
                        index2 += 1
    # third direction
    elif (binary_chromosome[18:21] == "010"):
        for h in range(int(binary_chromosome[2:10], 2), height):
            for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(
                        transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if (index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(
                            transformed[h][w], delimiter[index2])
                        index2 += 1
    # fourth direction
    elif (binary_chromosome[18:21] == "011"):
        for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
            for h in range(int(binary_chromosome[2:10], 2), height):
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(
                        transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if (index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(
                            transformed[h][w], delimiter[index2])
                        index2 += 1
    # fifth direction
    elif (binary_chromosome[18:21] == "100"):
        for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
            for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(
                        transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if (index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(
                            transformed[h][w], delimiter[index2])
                        index2 += 1
    # sixth direction
    elif (binary_chromosome[18:21] == "101"):
        for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
            for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(
                        transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if (index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(
                            transformed[h][w], delimiter[index2])
                        index2 += 1
    # seventh direction
    elif (binary_chromosome[18:21] == "110"):
        for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
            for w in range(int(binary_chromosome[10:18], 2), width):
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(
                        transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if (index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(
                            transformed[h][w], delimiter[index2])
                        index2 += 1
    # eigth direction
    elif (binary_chromosome[18:21] == "111"):
        for w in range(int(binary_chromosome[10:18], 2), width):
            for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
                if (index < len(binary_message)):
                    transformed[h][w] = hide_bit(
                        transformed[h][w], binary_message[index])
                    index += 1
                elif (index < len(binary_message) + len(delimiter)):
                    if (index2 < len(delimiter)):
                        transformed[h][w] = hide_bit(
                            transformed[h][w], delimiter[index2])
                        index2 += 1
    return transformed


def Decode(host, binary_chromosome):
    global delimiter
    height, width = host.shape
    binary_data = ""
    # first direction
    if (binary_chromosome[18:21] == "000"):
        for h in range(int(binary_chromosome[2:10], 2), height):
            for w in range(int(binary_chromosome[10:18], 2), width):
                if (binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w], 'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    # second direction
    elif (binary_chromosome[18:21] == "001"):
        for w in range(int(binary_chromosome[10:18], 2), width):
            for h in range(int(binary_chromosome[2:10], 2), height):
                if (binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w], 'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    # third direction
    elif (binary_chromosome[18:21] == "010"):
        for h in range(int(binary_chromosome[2:10], 2), height):
            for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
                if (binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w], 'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    # fourth direction
    elif (binary_chromosome[18:21] == "011"):
        for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
            for h in range(int(binary_chromosome[2:10], 2), height):
                if (binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w], 'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    # fifth direction
    elif (binary_chromosome[18:21] == "100"):
        for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
            for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
                if (binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w], 'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    # sixth direction
    elif (binary_chromosome[18:21] == "101"):
        for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
            for w in range(width - 1 - int(binary_chromosome[10:18], 2), 0, -1):
                if (binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w], 'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    # seventh direction
    elif (binary_chromosome[18:21] == "110"):
        for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
            for w in range(int(binary_chromosome[10:18], 2), width):
                if (binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w], 'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break
    # eigth direction
    elif (binary_chromosome[18:21] == "111"):
        for w in range(int(binary_chromosome[10:18], 2), width):
            for h in range(height - 1 - int(binary_chromosome[2:10], 2), 0, -1):
                if (binary_data[-40:] != delimiter):
                    bin_pixel_str = format(host[h][w], 'b')
                    bin_pixel = list(bin_pixel_str)
                    binary_data += bin_pixel[len(bin_pixel) - 1]
                else:
                    break

    return binary_data[:-40]

correct_counter = 0
def fitness_function(binary_chromosome):
    global image, secret_image, correct_counter, delimiter
    message = "A3FZ9YBQ" * 100  # 2 MB of 'A'

    #binary_data = ''.join(format(ord(c), '08b') for c in message)
    binary_data = ''.join(random.choice(['0', '1']) for _ in range(131072))
    secret = Encode(image, binary_data, binary_chromosome)
    decoded_data = Decode(secret, binary_chromosome)
    original_binary_data = image_to_binary_data(image)
    secret_binary_data = image_to_binary_data(secret)
    if (binary_data == decoded_data):correct_counter+=1
    psnr = cv2.PSNR(image, secret)
    ber = calculate_BER(original_binary_data, secret_binary_data)
    return psnr, ber

# Modified genetic and pso functions to track performance
# Function to convert an image to binary data


def image_to_binary_data(image):
    pixels = np.array(image).flatten()
    binary_data = ''.join(format(pixel, '08b') for pixel in pixels)
    return binary_data

def genetic_with_tracking():
    global best_solution
    population_size = 50
    num_generations = 100
    mutation_rate = 0.3
    population = [create_chromosome() for _ in range(population_size)]
    psnr_values = []
    ber_values = []

    for generation in range(num_generations):
        # Calculate fitness scores and store them for tracking
        fitness_scores = [fitness_function(x)[0] for x in population]
        # Track the highest PSNR for this generation
        psnr_values.append(max(fitness_scores))
        best_individual = population[np.argmax(fitness_scores)]
        # Track BER of the best individual
        ber_values.append(fitness_function(best_individual)[1])

        # Select parents for the next generation based on their fitness
        selected_parents = random.choices(
            population, weights=fitness_scores, k=population_size)

        # Create offspring using crossover
        offspring = []
        for i in range(0, population_size, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            crossover_point = random.randint(1, 21 - 1)
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            offspring.extend([offspring1, offspring2])

        # Apply mutation
        for i in range(population_size):
            for j in range(21):
                if random.random() < mutation_rate:
                    offspring[i] = offspring[i][:j] + \
                        ("0" if offspring[i][j] ==
                         "1" else "1") + offspring[i][j+1:]

        # Add the best individual of the current generation to the new offspring (promotion)
        best_fitness_index = np.argmax(fitness_scores)
        best_chromosome = population[best_fitness_index]

        # Ensure best individual is part of the new population
        # Replace the worst individual with the best if necessary
        # Place the best individual in the first position
        offspring[0] = best_chromosome
        population = offspring  # Set new population for the next generation

    # After all generations, the best solution found is the one with the highest fitness score
    best_solution = max(population, key=lambda x: fitness_function(x)[0])
    print(f"Genetic : {best_solution}")
    return psnr_values, ber_values


def pso_with_tracking():
    num_particles = 50
    num_dimensions = 21  # Number of bits in each binary string (chromosome)
    max_iterations = 100
    c1, c2 = 2.0, 2.0
    global_best_position = None

    # Initialize particles with random binary strings (0s and 1s)
    particles = [create_chromosome() for _ in range(num_particles)]
    velocities = [[random.uniform(-1, 1) for _ in range(num_dimensions)]
                  for _ in range(num_particles)]
    best_positions = particles[:]
    psnr_values = []
    ber_values = []

    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Calculate fitness for the current particle
            fitness = fitness_function(particles[i])[0]

            # Update the personal best if fitness improves
            if fitness_function(best_positions[i])[0] < fitness:
                best_positions[i] = particles[i]

            # Update the global best if fitness improves
            if global_best_position is None or fitness_function(global_best_position)[0] < fitness:
                global_best_position = particles[i]

        # Record PSNR and BER for the global best solution
        psnr_values.append(fitness_function(global_best_position)[0])
        ber_values.append(fitness_function(global_best_position)[1])

        # Update velocities and positions for each particle
        for i in range(num_particles):
            for j in range(num_dimensions):
                r1, r2 = random.random(), random.random()
                # Cognitive and social components of velocity update
                cognitive = c1 * r1 * \
                    (int(best_positions[i][j]) - int(particles[i][j]))
                social = c2 * r2 * \
                    (int(global_best_position[j]) - int(particles[i][j]))
                velocities[i][j] += cognitive + social

                # Update position: flip the bit with probability derived from the velocity
                # Update the bit position based on the sign of the velocity
                # Apply sigmoid to decide flip probability
                if random.random() < 1 / (1 + np.exp(-velocities[i][j])):
                    # Flip the bit
                    particles[i] = particles[i][:j] + \
                        ('1' if particles[i][j] ==
                         '0' else '0') + particles[i][j+1:]

    print(f"PSO : {global_best_position}")

    return psnr_values, ber_values



lock = Lock()

# Function Definitions


def run_genetic():
    global genetic_times, genetic_psnr, genetic_ber
    print("Starting Genetic Algorithm")
    try:
        start_time = time.time()
        g_psnr, g_ber = genetic_with_tracking()
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
        p_psnr, p_ber = pso_with_tracking()
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

    # Count bit errors
    bit_errors = sum(o != r for o, r in zip(original, received))

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
print(f"Correct : {correct_counter}")