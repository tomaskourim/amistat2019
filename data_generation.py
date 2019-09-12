# code used to generate data using theoretical models.

import logging
from typing import List

import numpy as np


def ising(bernoulli: int) -> int:
    """
    Transfers variable form Bernoulli random to Ising random; see
    https://en.wikipedia.org/wiki/Bernoulli_distribution
    https://en.wikipedia.org/wiki/Ising_model
    """
    if bernoulli == 1:
        return 1
    elif bernoulli == 0:
        return -1
    else:
        raise Exception('Unexpected value of Bernoulli distribution')


def generate_rw_success_punished(starting_probability: float, c_lambda: float, walk_steps: int, repetitions: int) -> \
        List[List[int]]:
    walks = []
    for j in range(0, repetitions):
        steps = ['']
        probabilities = [starting_probability]
        for i in range(1, walk_steps + 1):
            steps.append(ising(np.random.binomial(1, probabilities[i - 1], 1)[0]))  # next step using actual probability
            probabilities.append(
                c_lambda * probabilities[i - 1] + 0.5 * (1 - c_lambda) * (1 - steps[i]))  # as in Definition
        walks.append(steps)

    return walks


def generate_rw_success_rewarded(starting_probability: float, c_lambda: float, walk_steps: int, repetitions: int) -> \
        List[List[int]]:
    walks = []
    for j in range(0, repetitions):
        steps = ['']
        probabilities = [starting_probability]
        for i in range(1, walk_steps + 1):
            steps.append(ising(np.random.binomial(1, probabilities[i - 1], 1)[0]))  # next step using actual probability
            probabilities.append(
                c_lambda * probabilities[i - 1] + 0.5 * (1 - c_lambda) * (1 + steps[i]))  # as in Definition
        walks.append(steps)
    return walks


def main():
    # try different Random Walk with Varying Transition Probabilities definitions
    # different lambdas, starting probability, number of steps, multiple times with same starting variables
    # save into .csv?

    c_lambda = 0.5
    p0 = 0.5
    steps = 10
    repetitions = 8

    walks = generate_rw_success_rewarded(p0, c_lambda, steps, repetitions)
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
