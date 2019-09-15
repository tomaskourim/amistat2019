# code used to analyze generalized data

import logging
import pickle
from os import listdir
from os.path import isfile, join
from typing import List

import numpy as np
import scipy.optimize as opt

from common import get_current_probability
from config import DATA_DIRNAME


# try different possible models
# find the best suitable model & parameter values
# compare with reality
# estimate p0 only from first steps
# first estimate p0, then get lambda
# get lambda and p0


def get_single_walk_likelihood(log_likelihood: float, c_lambdas: List[float], starting_probability: float,
                               walk: List[int], walk_type: str, starting_index: int) -> float:
    current_probability = starting_probability
    for i in range(starting_index, len(walk)):
        current_probability = get_current_probability(c_lambdas, current_probability, walk[i - 1], walk_type)
        log_likelihood = log_likelihood + 0.5 * ((1 + walk[i]) * np.log(current_probability) + (1 - walk[i]) * np.log(
            1 - current_probability))
    return log_likelihood


def get_multiple_walks_log_likelihood(c_lambdas: List[float], starting_probability: float, walks: List[List[int]],
                                      walk_type: str, starting_index: int):
    log_likelihood = 0
    for walk in walks:
        log_likelihood = get_single_walk_likelihood(log_likelihood, c_lambdas, starting_probability, walk, walk_type,
                                                    starting_index)
    return log_likelihood


def negative_log_likelihood_single_lambda(c_lambda: float, walk_type: str, starting_probability: float,
                                          walks: List[List[int]]) -> float:
    starting_index = 2
    log_likelihood = get_multiple_walks_log_likelihood([c_lambda], starting_probability, walks, walk_type,
                                                       starting_index)
    return -log_likelihood


def negative_log_likelihood_multiple_lambda(c_lambdas: List[float], walk_type: str, starting_probability: float,
                                            walks: List[List[int]]) -> float:
    starting_index = 2
    log_likelihood = get_multiple_walks_log_likelihood(c_lambdas, starting_probability, walks, walk_type,
                                                       starting_index)
    return -log_likelihood


def negative_log_likelihood_p0(starting_probability: float, walk_type: str, c_lambdas: List[float],
                               walks: List[List[int]]) -> float:
    starting_index = 1
    log_likelihood = get_multiple_walks_log_likelihood(c_lambdas, starting_probability, walks, walk_type,
                                                       starting_index)
    return -log_likelihood


# p0 known, get lambda
def get_lambda_estimate(walk_type: str, starting_probability: float, walks: List[List[int]]):
    opt_result = opt.minimize_scalar(negative_log_likelihood_single_lambda, bounds=(0, 1), method='bounded',
                                     args=(walk_type, starting_probability, walks))
    if opt_result.success:
        logging.info("Fitted successfully.")
        return opt_result.x
    else:
        return None
    pass


def get_lambdas_estimate(walk_type, starting_probability, walks):
    guess = np.array([0.5, 0.5])
    opt_result = opt.minimize(negative_log_likelihood_multiple_lambda, guess, method='Nelder-Mead',
                              args=(walk_type, starting_probability, walks))
    if opt_result.success:
        logging.info("Fitted successfully.")
        return opt_result.x
    else:
        return None


# lambda known, get p0
def get_p0_estimate(walk_type: str, c_lambdas: List[float], walks: List[List[int]]):
    opt_result = opt.minimize_scalar(negative_log_likelihood_p0, bounds=(0, 1), method='bounded',
                                     args=(walk_type, c_lambdas, walks))
    if opt_result.success:
        logging.info("Fitted successfully.")
        return opt_result.x
    else:
        return None
    pass


def main():
    generated_data = [f for f in listdir(DATA_DIRNAME) if isfile(join(DATA_DIRNAME, f))]
    i = 0
    for datafile in generated_data:  # iterate over all generated cases
        with open(join(DATA_DIRNAME, datafile), 'rb') as f:
            walks, walk_type, starting_probability, c_lambdas, step_count = pickle.load(f)  # load data
            if walk_type == 'success_punished':
                continue
                # estimated_lambda = get_lambda_estimate(walk_type, starting_probability, walks)
                # if abs(c_lambdas[0] - estimated_lambda > 0.01):
                #     i = i + 1
                #     print(i, starting_probability, step_count, estimated_lambda, c_lambdas)

            elif walk_type == 'success_rewarded':
                continue
            elif walk_type == 'success_punished_two_lambdas':
                for walk in walks:
                    estimated_lambdas = get_lambdas_estimate(walk_type, starting_probability, [walk])

                    if max(estimated_lambdas) >= 1 or min(estimated_lambdas) <= 0:
                        print(i, starting_probability, step_count, estimated_lambdas, c_lambdas)
                i = i + 1
                print(i)
                # continue

            elif walk_type == 'success_rewarded_two_lambdas':
                continue
                # estimated_p0 = get_p0_estimate(walk_type, c_lambdas, walks)
                # if abs(starting_probability - estimated_p0 > 0.01):
                #     i = i + 1
                #     print(i, starting_probability, estimated_p0, step_count, c_lambdas)
            else:
                raise Exception(f'Unexpected walk type: {walk_type}')


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
