# code used to generate data using theoretical models.

import logging
from typing import List

import numpy as np


def ising(bernoulli: int) -> int:
    """
    Transfers variable form Bernoulli random to Ising random; see
    https://en.wikipedia.org/wiki/Bernoulli_distribution
    https://en.wikipedia.org/wiki/Ising_model

    :param bernoulli: int
    :return:
    """
    if bernoulli == 1:
        return 1
    elif bernoulli == 0:
        return -1
    else:
        raise Exception(f'Unexpected value of Bernoulli distribution: {bernoulli}')


def next_probability(walk_type: str, c_lambdas: List[float], steps: List[int], probabilities: List[float],
                     i: int) -> float:
    """
        Computes the next step transition probability according to the definition of given :param walk_type
    :param walk_type: str
    :param c_lambdas: List[float]
    :param steps: List[int]
    :param probabilities: List[float]
    :param i: int
    :return:
    """
    if walk_type == 'success_punished':
        return c_lambdas[0] * probabilities[i - 1] + 0.5 * (1 - c_lambdas[0]) * (1 - steps[i])
    elif walk_type == 'success_rewarded':
        return c_lambdas[0] * probabilities[i - 1] + 0.5 * (1 - c_lambdas[0]) * (1 + steps[i])
    elif walk_type == 'success_punished_two_lambdas':
        return 0.5 * ((1 + steps[i]) * c_lambdas[0] * probabilities[i - 1] + (1 - steps[i]) * (
                1 - c_lambdas[1] * (1 - probabilities[i - 1])))
    elif walk_type == 'success_rewarded_two_lambdas':
        return 0.5 * ((1 - steps[i]) * c_lambdas[0] * probabilities[i - 1] + (1 + steps[i]) * (
                1 - c_lambdas[1] * (1 - probabilities[i - 1])))
    else:
        raise Exception(f'Unexpected walk type: {walk_type}')


def generate_rw(walk_type: str, starting_probability: float, c_lambdas: List[float], walk_steps: int,
                repetitions: int) -> \
        List[List[int]]:
    walks = []
    for j in range(0, repetitions):
        steps = ['']
        probabilities = [starting_probability]
        for i in range(1, walk_steps + 1):
            steps.append(ising(np.random.binomial(1, probabilities[i - 1], 1)[0]))  # next step using actual probability
            probabilities.append(next_probability(walk_type, c_lambdas, steps, probabilities, i))
        walks.append(steps)
        print(probabilities)
        print(steps)
    return walks


def main():
    # try different Random Walk with Varying Transition Probabilities definitions
    # different lambdas, starting probability, number of steps, multiple times with same starting variables
    # iterate over lambdas, starting probability, number of steps (reward might lead to floating point errors),
    # repetitions
    # save into .csv?

    c_lambdas = [0.5]
    p0 = 0.5
    steps = 10
    repetitions = 3

    # walks = generate_rw('success_punished', p0, c_lambdas, steps, repetitions)
    # walks = generate_rw('success_rewarded', p0, c_lambdas, steps, repetitions)
    c_lambdas = [0.5, 0.1]
    # walks = generate_rw('success_punished_two_lambdas', p0, c_lambdas, steps, repetitions)
    walks = generate_rw('success_rewarded_two_lambdas', p0, c_lambdas, steps, repetitions)
    print(walks)


if __name__ == '__main__':
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel('DEBUG')

    # Create handlers
    total_handler = logging.FileHandler('logfile_total.log', mode='w')
    info_handler = logging.FileHandler('logfile_info.log')
    error_handler = logging.FileHandler('logfile_error.log')
    stdout_handler = logging.StreamHandler()

    total_handler.setLevel(logging.DEBUG)
    info_handler.setLevel(logging.INFO)
    error_handler.setLevel(logging.WARNING)
    stdout_handler.setLevel(logging.WARNING)

    # Create formatters and add it to handlers
    logging_format = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(message)s')
    total_handler.setFormatter(logging_format)
    info_handler.setFormatter(logging_format)
    error_handler.setFormatter(logging_format)
    stdout_handler.setFormatter(logging_format)

    # Add handlers to the logger
    logger.addHandler(total_handler)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(stdout_handler)

    main()
