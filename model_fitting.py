# code used fit models onto generalized data

import logging
import pickle
import sys
from datetime import datetime
from os import listdir
from os.path import isfile, join
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize as opt

from common import get_current_probability
from config import DATA_DIRNAME, OPTIMIZATION_ALGORITHM, MODEL_PARAMETERS, PREDICTION_VALUES, REPETITIONS_OF_WALK_S, \
    REPETITIONS_OF_WALK_SERIES, ERROR_VALUE, BASE_GUESSES


# compare with reality
# TODO estimate p0 only from first steps
# TODO first estimate p0, then get lambda


def get_single_walk_log_likelihood(log_likelihood: float, c_lambdas: List[float], starting_probability: float,
                                   walk: List[int], model_type: str, starting_index: int) -> float:
    if starting_probability >= 1 or starting_probability <= 0 or max(c_lambdas) >= 1 or min(c_lambdas) <= 0:
        return -(sys.float_info.max / 2 - 5)  # so that I dont get double overflow error
    current_probability = starting_probability
    for i in range(starting_index, len(walk)):
        current_probability = get_current_probability(c_lambdas, current_probability, walk[i - 1], model_type)
        log_likelihood = log_likelihood + 0.5 * ((1 + walk[i]) * np.log(current_probability) + (1 - walk[i]) * np.log(
            1 - current_probability))
    return log_likelihood


def get_multiple_walks_log_likelihood(c_lambdas: List[float], starting_probability: float, walks: List[List[int]],
                                      model_type: str, starting_index: int):
    log_likelihood = 0
    for walk in walks:
        log_likelihood = get_single_walk_log_likelihood(log_likelihood, c_lambdas, starting_probability, walk,
                                                        model_type, starting_index)
    return log_likelihood


def negative_log_likelihood_single_lambda(c_lambda: float, model_type: str, starting_probability: float,
                                          walks: List[List[int]]) -> float:
    starting_index = 2
    log_likelihood = get_multiple_walks_log_likelihood([c_lambda], starting_probability, walks, model_type,
                                                       starting_index)
    return -log_likelihood


def negative_log_likelihood_multiple_lambda(c_lambdas: List[float], model_type: str, starting_probability: float,
                                            walks: List[List[int]]) -> float:
    starting_index = 2
    log_likelihood = get_multiple_walks_log_likelihood(c_lambdas, starting_probability, walks, model_type,
                                                       starting_index)
    return -log_likelihood


def negative_log_likelihood_p0(starting_probability: float, model_type: str, c_lambdas: List[float],
                               walks: List[List[int]]) -> float:
    starting_index = 1
    log_likelihood = get_multiple_walks_log_likelihood(c_lambdas, starting_probability, walks, model_type,
                                                       starting_index)
    return -log_likelihood


def negative_log_likelihood_params(params: List[float], model_type: str, walks: List[List[int]]) -> float:
    starting_probability = params[0]
    c_lambdas = params[1:]
    starting_index = 1
    log_likelihood = get_multiple_walks_log_likelihood(c_lambdas, starting_probability, walks, model_type,
                                                       starting_index)
    return -log_likelihood


# p0 known, get lambda
def get_lambda_estimate(model_type: str, starting_probability: float, walks: List[List[int]],
                        base_guess: float) -> Union[float, List[float]]:
    if model_type == 'success_punished' or model_type == 'success_rewarded':
        error_value = ERROR_VALUE
        opt_result = opt.minimize_scalar(negative_log_likelihood_single_lambda, bounds=(0, 1), method='bounded',
                                         args=(model_type, starting_probability, walks))
    elif model_type == 'success_punished_two_lambdas' or model_type == 'success_rewarded_two_lambdas':
        guess = np.repeat(base_guess, 2)
        bounds = opt.Bounds((0, 0), (1, 1), keep_feasible=True)
        error_value = np.repeat(ERROR_VALUE, len(guess))
        opt_result = opt.minimize(negative_log_likelihood_multiple_lambda, guess, method=OPTIMIZATION_ALGORITHM,
                                  args=(model_type, starting_probability, walks))
    else:
        raise Exception(f'Unexpected walk type: {model_type}')

    if opt_result.success:
        logging.debug("Fitted successfully.")
        return opt_result.x
    else:
        return error_value


# lambda known, get p0
def get_p0_estimate(model_type: str, c_lambdas: List[float], walks: List[List[int]]) -> float:
    opt_result = opt.minimize_scalar(negative_log_likelihood_p0, bounds=(0, 1), method='bounded',
                                     args=(model_type, c_lambdas, walks))
    if opt_result.success:
        logging.debug("Fitted successfully.")
        return opt_result.x
    else:
        return ERROR_VALUE


# get lambda and p0
def get_parameters_estimate(model_type: str, walks: List[List[int]], base_guess: float) -> Union[List[float], np.array]:
    if model_type == 'success_punished' or model_type == 'success_rewarded':
        guess = np.repeat(base_guess, 2)
        bounds = opt.Bounds((0, 0), (1, 1), keep_feasible=True)
    elif model_type == 'success_punished_two_lambdas' or model_type == 'success_rewarded_two_lambdas':
        guess = np.repeat(base_guess, 3)
        bounds = opt.Bounds((0, 0, 0), (1, 1, 1), keep_feasible=True)
    else:
        raise Exception(f'Unexpected walk type: {model_type}')

    opt_result = opt.minimize(negative_log_likelihood_params, guess, method=OPTIMIZATION_ALGORITHM,
                              args=(model_type, walks))
    if opt_result.success:
        logging.debug("Fitted successfully.")
        return opt_result.x
    else:
        return np.repeat(ERROR_VALUE, len(guess))


def find_akaike(guess: np.ndarray, model: str, walks: List[List[int]], result: List[float], current_model: str,
                min_akaike: float, bounds: opt.Bounds) -> Tuple[float, List[float], str]:
    opt_result = opt.minimize(negative_log_likelihood_params, guess, method=OPTIMIZATION_ALGORITHM,
                              args=(model, walks))
    akaike = 2 * len(guess) + 2 * opt_result.fun
    if opt_result.success and akaike < min_akaike:
        min_akaike = akaike
        current_model = model
        result = opt_result.x
    return min_akaike, result, current_model


# find the best suitable model & parameter values
def get_model_estimate(walks: List[List[int]], base_guess: float) -> Tuple[List[float], str]:
    """
    Uses the Akaike information criterion
    https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size
    to get the optimal model.
    AIC = 2k - 2ln(L)
    opt.minimize(negative_log_likelihood_params, guess, method='Nelder-Mead', args=(model, walks)) returns directly
    - ln(L)
    :param walks:
    :param base_guess:
    :return:
    """
    result = [ERROR_VALUE, ERROR_VALUE]
    current_model = ERROR_VALUE
    min_akaike = sys.float_info.max

    # single lambda models
    guess = np.repeat(base_guess, 2)
    bounds = opt.Bounds((0, 0), (1, 1), keep_feasible=True)
    model = 'success_punished'
    min_akaike, result, current_model = find_akaike(guess, model, walks, result, current_model, min_akaike, bounds)

    model = 'success_rewarded'
    min_akaike, result, current_model = find_akaike(guess, model, walks, result, current_model, min_akaike, bounds)

    # two lambdas models
    guess = np.repeat(base_guess, 3)
    bounds = opt.Bounds((0, 0, 0), (1, 1, 1), keep_feasible=True)
    model = 'success_punished_two_lambdas'
    min_akaike, result, current_model = find_akaike(guess, model, walks, result, current_model, min_akaike, bounds)

    model = 'success_rewarded_two_lambdas'
    min_akaike, result, current_model = find_akaike(guess, model, walks, result, current_model, min_akaike, bounds)

    return result, current_model


def estimate_current_walks_lambda(walks: List[List[int]], model_type: str, starting_probability: float,
                                  basic_result: dict) -> dict:
    estimated_lambda, estimated_lambda0, estimated_lambda1 = ERROR_VALUE, ERROR_VALUE, ERROR_VALUE
    current_result = basic_result.copy()
    for base_guess in BASE_GUESSES:
        estimated_lambdas = get_lambda_estimate(model_type, starting_probability, walks, base_guess)
        if model_type == 'success_punished' or model_type == 'success_rewarded':
            estimated_lambda = estimated_lambdas
            estimated_lambda0 = ""
            estimated_lambda1 = ""
        elif model_type == 'success_punished_two_lambdas' or model_type == 'success_rewarded_two_lambdas':
            estimated_lambda = ""
            estimated_lambda0 = estimated_lambdas[0]
            estimated_lambda1 = estimated_lambdas[1]
        else:
            raise Exception(f'Unexpected walk type: {model_type}')
        if estimated_lambda != ERROR_VALUE and estimated_lambda0 != ERROR_VALUE and estimated_lambda1 != ERROR_VALUE:
            break
    current_result["prediction_type"] = "only_lambda"
    current_result["predicted_lambda"] = estimated_lambda
    current_result["predicted_lambda0"] = estimated_lambda0
    current_result["predicted_lambda1"] = estimated_lambda1
    return current_result


def estimate_current_walks_p0(walks: List[List[int]], model_type: str, c_lambdas: List[float],
                              basic_result: dict) -> dict:
    current_result = basic_result.copy()
    estimated_p0 = get_p0_estimate(model_type, c_lambdas, walks)
    current_result["prediction_type"] = "only_p0"
    current_result["predicted_p0"] = estimated_p0
    return current_result


def estimate_current_walks_all(walks: List[List[int]], model_type: str, basic_result: dict) -> dict:
    estimated_lambda, estimated_lambda0, estimated_lambda1, estimated_p0 = ERROR_VALUE, ERROR_VALUE, ERROR_VALUE, ERROR_VALUE
    current_result = basic_result.copy()
    for base_guess in BASE_GUESSES:
        estimated_params = get_parameters_estimate(model_type, walks, base_guess)
        estimated_p0 = estimated_params[0]
        if model_type == 'success_punished' or model_type == 'success_rewarded':
            estimated_lambda = estimated_params[1]
            estimated_lambda0 = ""
            estimated_lambda1 = ""
        elif model_type == 'success_punished_two_lambdas' or model_type == 'success_rewarded_two_lambdas':
            estimated_lambda = ""
            estimated_lambda0 = estimated_params[1]
            estimated_lambda1 = estimated_params[2]
        else:
            raise Exception(f'Unexpected walk type: {model_type}')
        if estimated_p0 != ERROR_VALUE and estimated_lambda != ERROR_VALUE and estimated_lambda0 != ERROR_VALUE \
                and estimated_lambda1 != ERROR_VALUE:
            break
    current_result["prediction_type"] = "all_parameters"
    current_result["predicted_lambda"] = estimated_lambda
    current_result["predicted_lambda0"] = estimated_lambda0
    current_result["predicted_lambda1"] = estimated_lambda1
    current_result["predicted_p0"] = estimated_p0
    return current_result


def estimate_current_walks_model(walks: List[List[int]], model_type: str, basic_result: dict) -> dict:
    estimated_params = [ERROR_VALUE, ERROR_VALUE]
    estimated_model = ERROR_VALUE
    current_result = basic_result.copy()
    for base_guess in BASE_GUESSES:
        estimated_params, estimated_model = get_model_estimate(walks, base_guess)
        if estimated_model != ERROR_VALUE:
            break
    if model_type == 'success_punished' or model_type == 'success_rewarded':
        estimated_lambda = estimated_params[1]
        estimated_lambda0 = ""
        estimated_lambda1 = ""
    elif model_type == 'success_punished_two_lambdas' or model_type == 'success_rewarded_two_lambdas':
        estimated_lambda = ""
        estimated_lambda0 = estimated_params[1]
        estimated_lambda1 = estimated_params[2] if len(estimated_params) == 3 else ERROR_VALUE
    else:
        raise Exception(f'Unexpected walk type: {model_type}')

    current_result["prediction_type"] = "everything"
    current_result["predicted_model"] = estimated_model
    current_result["predicted_lambda"] = estimated_lambda
    current_result["predicted_lambda0"] = estimated_lambda0
    current_result["predicted_lambda1"] = estimated_lambda1
    current_result["predicted_p0"] = estimated_params[0]
    return current_result


def get_basic_result(model_type: str, c_lambda: float, c_lambda0: float, c_lambda1: float, starting_probability: float,
                     step_count: int, repetition: int) -> dict:
    basic_result = {"model_type": model_type,
                    "c_lambda": c_lambda,
                    "c_lambda0": c_lambda0,
                    "c_lambda1": c_lambda1,
                    "p0": starting_probability,
                    "step_count": step_count,
                    "prediction_type": "",
                    "predicted_model": "",
                    "predicted_lambda": "",
                    "predicted_lambda0": "",
                    "predicted_lambda1": "",
                    "predicted_p0": "",
                    "repetition": repetition}
    return basic_result


def main():
    generated_data = [f for f in listdir(f"{DATA_DIRNAME}/K{REPETITIONS_OF_WALK_S[0]}") if
                      isfile(join(DATA_DIRNAME, f"K{REPETITIONS_OF_WALK_S[0]}", f))]
    columns = MODEL_PARAMETERS
    columns.append("prediction_type")
    columns.extend(PREDICTION_VALUES)
    columns.append("repetition")
    results = pd.DataFrame(columns=columns)
    start_time_loop = datetime.now()
    iterations = len(generated_data)
    for i, datafile in enumerate(generated_data):  # iterate over all generated cases
        start_time_iter = datetime.now()
        with open(join(DATA_DIRNAME, f"K{REPETITIONS_OF_WALK_S[0]}", datafile), 'rb') as f:
            walks, model_type, starting_probability, c_lambdas, step_count, repetition = pickle.load(f)  # load data
        if model_type == 'success_punished' or model_type == 'success_rewarded':
            c_lambda = c_lambdas[0]
            c_lambda0 = ""
            c_lambda1 = ""
        elif model_type == 'success_punished_two_lambdas' or model_type == 'success_rewarded_two_lambdas':
            c_lambda = ""
            c_lambda0 = c_lambdas[0]
            c_lambda1 = c_lambdas[1]
        else:
            raise Exception(f'Unexpected walk type: {model_type}')
        basic_result = get_basic_result(model_type, c_lambda, c_lambda0, c_lambda1, starting_probability, step_count,
                                        repetition)
        # TODO here are actually just walk steps but want to use complete walks
        current_result = estimate_current_walks_lambda(walks, model_type, starting_probability, basic_result)
        results = results.append(current_result, ignore_index=True)

        current_result = estimate_current_walks_p0(walks, model_type, c_lambdas, basic_result)
        results = results.append(current_result, ignore_index=True)

        current_result = estimate_current_walks_all(walks, model_type, basic_result)
        results = results.append(current_result, ignore_index=True)

        current_result = estimate_current_walks_model(walks, model_type, basic_result)
        results = results.append(current_result, ignore_index=True)

        end_time_iter = datetime.now()
        time_curr = end_time_iter - start_time_iter
        time_per_iter = (end_time_iter - start_time_loop) / (i + 1)
        eta = (iterations - i - 1) * time_per_iter
        logging.info(
            f"{time_curr}; AVG {time_per_iter}; ETA: {eta}. "
            f"{iterations - i - 1}/{iterations}; {datafile}")

    with open(f"results_{OPTIMIZATION_ALGORITHM}_K{REPETITIONS_OF_WALK_S[0]}_N{REPETITIONS_OF_WALK_SERIES}.pkl",
              'wb') as f:
        pickle.dump([results], f)


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
    stdout_handler.setLevel(logging.INFO)

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
    logging.info(f"Duration: {(end_time - start_time)}")
