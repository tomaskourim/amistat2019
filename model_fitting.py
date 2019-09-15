# code used to analyze generalized data

import logging
import pickle
import sys
from datetime import datetime
from os import listdir
from os.path import isfile, join
from typing import List

import numpy as np
import scipy.optimize as opt

from common import get_current_probability
from config import DATA_DIRNAME


# compare with reality
# TODO estimate p0 only from first steps
# TODO first estimate p0, then get lambda


def get_single_walk_log_likelihood(log_likelihood: float, c_lambdas: List[float], starting_probability: float,
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
        log_likelihood = get_single_walk_log_likelihood(log_likelihood, c_lambdas, starting_probability, walk,
                                                        walk_type,
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


def negative_log_likelihood_params(params: List[float], walk_type: str, walks: List[List[int]]) -> float:
    starting_probability = params[0]
    c_lambdas = params[1:]
    starting_index = 1
    log_likelihood = get_multiple_walks_log_likelihood(c_lambdas, starting_probability, walks, walk_type,
                                                       starting_index)
    return -log_likelihood


# p0 known, get lambda
def get_lambda_estimate(walk_type: str, starting_probability: float, walks: List[List[int]]):
    if walk_type == 'success_punished' or walk_type == 'success_rewarded':
        opt_result = opt.minimize_scalar(negative_log_likelihood_single_lambda, bounds=(0, 1), method='bounded',
                                         args=(walk_type, starting_probability, walks))
    elif walk_type == 'success_punished_two_lambdas' or walk_type == 'success_rewarded_two_lambdas':
        guess = np.array([0.5, 0.5])
        opt_result = opt.minimize(negative_log_likelihood_multiple_lambda, guess, method='Nelder-Mead',
                                  args=(walk_type, starting_probability, walks))
    else:
        raise Exception(f'Unexpected walk type: {walk_type}')

    if opt_result.success:
        logging.info("Fitted successfully.")
        return opt_result.x
    else:
        return None
    pass


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


# get lambda and p0
def get_parameters_estimate(walk_type, walks):
    if walk_type == 'success_punished' or walk_type == 'success_rewarded':
        guess = np.array([0.5, 0.5])
    elif walk_type == 'success_punished_two_lambdas' or walk_type == 'success_rewarded_two_lambdas':
        guess = np.array([0.5, 0.5, 0.5])
    else:
        raise Exception(f'Unexpected walk type: {walk_type}')

    opt_result = opt.minimize(negative_log_likelihood_params, guess, method='Nelder-Mead', args=(walk_type, walks))
    if opt_result.success:
        logging.info("Fitted successfully.")
        return opt_result.x
    else:
        return None


# find the best suitable model & parameter values
def get_model_estimate(walks):
    result = None
    current_model = ''
    min_log_likelihood = sys.float_info.max

    # single lambda models
    guess = np.array([0.5, 0.5])
    model = 'success_punished'
    opt_result = opt.minimize(negative_log_likelihood_params, guess, method='Nelder-Mead', args=(model, walks))
    if opt_result.success and opt_result.fun < min_log_likelihood:
        min_log_likelihood = opt_result.fun
        current_model = model
        result = opt_result.x

    model = 'success_rewarded'
    opt_result = opt.minimize(negative_log_likelihood_params, guess, method='Nelder-Mead', args=(model, walks))
    if opt_result.success and opt_result.fun < min_log_likelihood:
        min_log_likelihood = opt_result.fun
        current_model = model
        result = opt_result.x

    # two lambdas models
    guess = np.array([0.5, 0.5, 0.5])
    model = 'success_punished_two_lambdas'
    opt_result = opt.minimize(negative_log_likelihood_params, guess, method='Nelder-Mead', args=(model, walks))
    if opt_result.success and opt_result.fun < min_log_likelihood:
        min_log_likelihood = opt_result.fun
        current_model = model
        result = opt_result.x

    model = 'success_rewarded_two_lambdas'
    opt_result = opt.minimize(negative_log_likelihood_params, guess, method='Nelder-Mead', args=(model, walks))
    if opt_result.success and opt_result.fun < min_log_likelihood:
        min_log_likelihood = opt_result.fun
        current_model = model
        result = opt_result.x

    return result, current_model


def main():
    generated_data = [f for f in listdir(DATA_DIRNAME) if isfile(join(DATA_DIRNAME, f))]
    start_time_loop = datetime.now()
    for i, datafile in enumerate(generated_data):  # iterate over all generated cases
        start_time_iter = datetime.now()
        print('---------------------------------------------------------')
        with open(join(DATA_DIRNAME, datafile), 'rb') as f:
            walks, walk_type, starting_probability, c_lambdas, step_count = pickle.load(f)  # load data
            if walk_type == 'success_punished':
                # print(i)
                # continue
                print(walk_type, step_count, starting_probability, c_lambdas)
                estimated_lambda = get_lambda_estimate(walk_type, starting_probability, walks)
                print(estimated_lambda)
                estimated_p0 = get_p0_estimate(walk_type, c_lambdas, walks)
                print(estimated_p0)
                estimated_params = get_parameters_estimate(walk_type, walks)
                print(estimated_params)
                estimated_params, estimated_model = get_model_estimate(walks)
                print(estimated_model, estimated_params)

            elif walk_type == 'success_rewarded':
                print(walk_type, step_count, starting_probability, c_lambdas)
                estimated_lambda = get_lambda_estimate(walk_type, starting_probability, walks)
                print(estimated_lambda)
                estimated_p0 = get_p0_estimate(walk_type, c_lambdas, walks)
                print(estimated_p0)
                estimated_params = get_parameters_estimate(walk_type, walks)
                print(estimated_params)
                estimated_params, estimated_model = get_model_estimate(walks)
                print(estimated_model, estimated_params)
                # continue
            elif walk_type == 'success_punished_two_lambdas':
                # print(i)
                # continue
                print(walk_type, step_count, starting_probability, c_lambdas)
                estimated_lambdas = get_lambda_estimate(walk_type, starting_probability, walks)
                print(estimated_lambdas)
                estimated_p0 = get_p0_estimate(walk_type, c_lambdas, walks)
                print(estimated_p0)
                estimated_params = get_parameters_estimate(walk_type, walks)
                print(estimated_params)
                estimated_params, estimated_model = get_model_estimate(walks)
                print(estimated_model, estimated_params)

            elif walk_type == 'success_rewarded_two_lambdas':
                # print(i)
                print(walk_type, step_count, starting_probability, c_lambdas)
                estimated_lambdas = get_lambda_estimate(walk_type, starting_probability, walks)
                print(estimated_lambdas)
                estimated_p0 = get_p0_estimate(walk_type, c_lambdas, walks)
                print(estimated_p0)
                estimated_params = get_parameters_estimate(walk_type, walks)
                print(estimated_params)
                estimated_params, estimated_model = get_model_estimate(walks)
                print(estimated_model, estimated_params)
                # continue
            else:
                raise Exception(f'Unexpected walk type: {walk_type}')
            end_time_iter = datetime.now()
            time_curr = end_time_iter - start_time_iter
            time_per_iter = (end_time_iter - start_time_loop) / (i + 1)
            eta = (len(generated_data) - i - 1) * time_per_iter
            print(f"Current iteration: {time_curr}. Per iter {time_per_iter}. ETA: {eta}")


if __name__ == '__main__':
    start_time = datetime.now()
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
    end_time = datetime.now()
    logging.info(f"\nDuration: {(end_time - start_time)}")
