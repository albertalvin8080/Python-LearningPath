import numpy as np

POP_SIZE = 4  # Population size.
NUM_GENERATIONS = 5  # Number of generations.
X_MM_DEC_VAL = 15  # Max/Min decimal value of x.
X_M_BITS = 4  # Number of bits (without the sign).
MUTATION_RATE = 0.05


def to_decimal(arr):
    x_decimal = 0
    for bit in arr[1:]:
        x_decimal = (x_decimal << 1) | bit
    sign = arr[0]
    x_decimal = x_decimal * -1 if sign == 1 else x_decimal
    return x_decimal


# x E [-15, +15]
def aptitude(x):
    x_decimal = x
    if isinstance(x, (np.ndarray, list)):
        x_decimal = to_decimal(x)
    return x_decimal**2 - 3 * x_decimal + 4


# Objective: lower values represent greater aptitude.
def aptitude_comparator(v1: int, v2: int):
    # 0: for v1 less than v2
    # 1: for v2 less or equals to v1
    return 0 if v1 < v2 else 1


def first_generation():
    gen = []
    for _ in range(POP_SIZE):
        sign = np.random.randint(0, 2)
        gen.append([sign] + [np.random.randint(0, 2) for _ in range(X_M_BITS)])

    return np.array(gen)


def tournament(population: np.ndarray):
    winners = []
    for _ in range(POP_SIZE):
        duelers = np.random.choice(POP_SIZE, size=2, replace=False)
        sinner_idx = aptitude_comparator(
            aptitude(population[duelers[0]]),
            aptitude(population[duelers[1]]),
        )
        winners.append(population[duelers[sinner_idx]])

    return np.array(winners)


def crossover(population: np.ndarray):
    new_pop = []
    for pair in [population[i : i + 2] for i in range(0, len(population), 2)]:
        children = crossover_step(pair[0], pair[1])
        new_pop.append(children[0])
        new_pop.append(children[1])
    return np.array(new_pop)


# ind* -> individual
def crossover_step(ind1: np.ndarray, ind2: np.ndarray):
    # +1 for the sign
    mask = np.random.randint(0, 2, size=X_M_BITS + 1)
    children = [[], []]
    for idx, mask_bit in enumerate(mask):
        ind1_bit = ind1[idx]
        ind2_bit = ind2[idx]
        children[0].append(ind1_bit if mask_bit == 1 else ind2_bit)
        children[1].append(ind2_bit if mask_bit == 0 else ind1_bit)
    return np.array(children)


def mutation(population: np.ndarray):
    original_shape = population.shape
    quant_of_mutations = int(MUTATION_RATE * population.size)
    reshaped_pop = population.reshape(population.size)

    for _ in range(0, quant_of_mutations):
        idx = np.random.randint(0, reshaped_pop.size)
        reshaped_pop[idx] = 0 if reshaped_pop[idx] == 1 else 1

    return reshaped_pop.reshape(original_shape)


def print_generation(population, n):
    print(f"========== Generation {n} ==========")
    print(f"A1: {to_decimal(population[0])}")
    print(f"A2: {to_decimal(population[1])}")
    print(f"A3: {to_decimal(population[2])}")
    print(f"A4: {to_decimal(population[3])}")


if __name__ == "__main__":
    fg = first_generation()
    print_generation(fg, 1)

    population = fg
    for n in range(2, NUM_GENERATIONS + 1):
        winners = tournament(population)
        population = crossover(population)
        population = mutation(population)
        print_generation(population, n)
