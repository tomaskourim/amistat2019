# code used to analyze generalized data

import logging
import pickle
import sys
from datetime import datetime
from os import listdir
from os.path import isfile, join
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as opt

from common import get_current_probability
from config import DATA_DIRNAME, CONFIDENCE_INTERVAL_SIZE


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
def get_lambda_estimate(walk_type: str, starting_probability: float, walks: List[List[int]]) -> float:
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
        logging.debug("Fitted successfully.")
        return opt_result.x
    else:
        if walk_type == 'success_punished' or walk_type == 'success_rewarded':
            return -10000
        elif walk_type == 'success_punished_two_lambdas' or walk_type == 'success_rewarded_two_lambdas':
            return [-10000, -10000]


# lambda known, get p0
def get_p0_estimate(walk_type: str, c_lambdas: List[float], walks: List[List[int]]) -> float:
    opt_result = opt.minimize_scalar(negative_log_likelihood_p0, bounds=(0, 1), method='bounded',
                                     args=(walk_type, c_lambdas, walks))
    if opt_result.success:
        logging.debug("Fitted successfully.")
        return opt_result.x
    else:
        return -10000


# get lambda and p0
def get_parameters_estimate(walk_type: str, walks: List[List[int]]) -> List[float]:
    if walk_type == 'success_punished' or walk_type == 'success_rewarded':
        guess = np.array([0.5, 0.5])
    elif walk_type == 'success_punished_two_lambdas' or walk_type == 'success_rewarded_two_lambdas':
        guess = np.array([0.5, 0.5, 0.5])
    else:
        raise Exception(f'Unexpected walk type: {walk_type}')

    opt_result = opt.minimize(negative_log_likelihood_params, guess, method='Nelder-Mead', args=(walk_type, walks))
    if opt_result.success:
        logging.debug("Fitted successfully.")
        return opt_result.x
    else:
        if walk_type == 'success_punished' or walk_type == 'success_rewarded':
            return [-10000, -10000]
        elif walk_type == 'success_punished_two_lambdas' or walk_type == 'success_rewarded_two_lambdas':
            return [-10000, -10000, -10000]


# find the best suitable model & parameter values
def get_model_estimate(walks: List[List[int]]) -> Tuple[List[float], str]:
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


def analyze_result_single_lambda(result: pd.DataFrame, prediction_type: str, model_type: str):
    lambdas = result['c_lambda'].unique()
    p0s = result['p0'].unique()
    step_sizes = result['step_count'].unique()
    tries = 0
    successes = 0
    for p0 in p0s:
        for step_size in step_sizes:
            for c_lambda in lambdas:
                result_of_test = result[
                    result.c_lambda.isin([c_lambda]) & result.p0.isin([p0]) & result.step_count.isin([step_size])]
                if len(result_of_test) > 1:
                    raise Exception(f"Result is supposed to be of size 1, but is of size {len(result)}")
                if prediction_type in ["only_lambda", "all_parameters"]:
                    tries = tries + 1
                    prediction = result_of_test.predicted_lambda.values[0]
                    if c_lambda * (1 - CONFIDENCE_INTERVAL_SIZE / 2) <= prediction <= c_lambda * (
                            1 + CONFIDENCE_INTERVAL_SIZE / 2):
                        successes = successes + 1
                if prediction_type in ["only_p0", "all_parameters"]:
                    tries = tries + 1
                    prediction = result_of_test.predicted_p0.values[0]
                    if p0 * (1 - CONFIDENCE_INTERVAL_SIZE / 2) <= prediction <= p0 * (
                            1 + CONFIDENCE_INTERVAL_SIZE / 2):
                        successes = successes + 1
                if prediction_type in ["everything"]:
                    tries = tries + 1
                    if result_of_test.predicted_model.values[0] == model_type:
                        successes = successes + 1
    print(
        f"Model type: {model_type}, prediction type: {prediction_type}. Successes: {successes}. Tries: {tries}. Success rate: {successes / tries}")


def analyze_result_multiple_lambda(result, prediction_type, model_type):
    lambdas0 = result['c_lambda0'].unique()
    lambdas1 = result['c_lambda1'].unique()
    p0s = result['p0'].unique()
    step_sizes = result['step_count'].unique()
    tries = 0
    successes = 0
    for p0 in p0s:
        for step_size in step_sizes:
            for c_lambda0 in lambdas0:
                for c_lambda1 in lambdas1:
                    result_of_test = result[
                        result.c_lambda0.isin([c_lambda0]) & result.c_lambda1.isin([c_lambda1]) & result.p0.isin(
                            [p0]) & result.step_count.isin([step_size])]
                    if len(result_of_test) == 0:
                        continue
                    if len(result_of_test) > 1:
                        raise Exception(f"Result is supposed to be of size 1, but is of size {len(result)}")
                    if prediction_type in ["only_lambda", "all_parameters"]:
                        tries = tries + 1
                        prediction = result_of_test.predicted_lambda0.values[0]
                        if c_lambda0 * (1 - CONFIDENCE_INTERVAL_SIZE / 2) <= prediction <= c_lambda0 * (
                                1 + CONFIDENCE_INTERVAL_SIZE / 2):
                            successes = successes + 1
                        tries = tries + 1
                        prediction = result_of_test.predicted_lambda1.values[0]
                        if c_lambda1 * (1 - CONFIDENCE_INTERVAL_SIZE / 2) <= prediction <= c_lambda1 * (
                                1 + CONFIDENCE_INTERVAL_SIZE / 2):
                            successes = successes + 1
                    if prediction_type in ["only_p0", "all_parameters"]:
                        tries = tries + 1
                        prediction = result_of_test.predicted_p0.values[0]
                        if p0 * (1 - CONFIDENCE_INTERVAL_SIZE / 2) <= prediction <= p0 * (
                                1 + CONFIDENCE_INTERVAL_SIZE / 2):
                            successes = successes + 1
                    if prediction_type in ["everything"]:
                        tries = tries + 1
                        if result_of_test.predicted_model.values[0] == model_type:
                            successes = successes + 1
    print(
        f"Model type: {model_type}, prediction type: {prediction_type}. Successes: {successes}. Tries: {tries}. Success rate: {successes / tries}")


def analyze_results(results: pd.DataFrame):
    model_types = results['model_type'].unique()
    prediction_types = results['prediction_type'].unique()
    for model_type in model_types:
        for prediction_type in prediction_types:
            if model_type == 'success_punished' or model_type == 'success_rewarded':
                analyze_result_single_lambda(
                    results[results.model_type.isin([model_type]) & results.prediction_type.isin([prediction_type])],
                    prediction_type, model_type)
            elif model_type == 'success_punished_two_lambdas' or model_type == 'success_rewarded_two_lambdas':
                analyze_result_multiple_lambda(
                    results[results.model_type.isin([model_type]) & results.prediction_type.isin([prediction_type])],
                    prediction_type, model_type)
            else:
                raise Exception(f'Unexpected walk type: {model_type}')


def main():
    generated_data = [f for f in listdir(DATA_DIRNAME) if isfile(join(DATA_DIRNAME, f))]
    results = pd.DataFrame(
        columns=["model_type", "c_lambda", "c_lambda0", "c_lambda1", "p0", "step_count", "prediction_type",
                 "predicted_model", "predicted_lambda", "predicted_lambda0", "predicted_lambda1",
                 "predicted_p0"])
    start_time_loop = datetime.now()
    # with open("results.pkl", 'rb') as f:
    #     results = pickle.load(f)  # load data
    for i, datafile in enumerate(generated_data):  # iterate over all generated cases
        start_time_iter = datetime.now()
        with open(join(DATA_DIRNAME, datafile), 'rb') as f:
            walks, walk_type, starting_probability, c_lambdas, step_count = pickle.load(f)  # load data
            print(walk_type, starting_probability, c_lambdas, step_count)
            if walk_type == 'success_punished' or walk_type == 'success_rewarded':
                c_lambda = c_lambdas[0]
                c_lambda0 = ""
                c_lambda1 = ""
            elif walk_type == 'success_punished_two_lambdas' or walk_type == 'success_rewarded_two_lambdas':
                c_lambda = ""
                c_lambda0 = c_lambdas[0]
                c_lambda1 = c_lambdas[1]
            else:
                raise Exception(f'Unexpected walk type: {walk_type}')

            estimated_lambdas = get_lambda_estimate(walk_type, starting_probability, walks)
            if walk_type == 'success_punished' or walk_type == 'success_rewarded':
                estimated_lambda = estimated_lambdas
                estimated_lambda0 = ""
                estimated_lambda1 = ""
            elif walk_type == 'success_punished_two_lambdas' or walk_type == 'success_rewarded_two_lambdas':
                estimated_lambda = ""
                estimated_lambda0 = estimated_lambdas[0]
                estimated_lambda1 = estimated_lambdas[1]
            else:
                raise Exception(f'Unexpected walk type: {walk_type}')
            current_result = {"model_type": walk_type,
                              "c_lambda": c_lambda,
                              "c_lambda0": c_lambda0,
                              "c_lambda1": c_lambda1,
                              "p0": starting_probability,
                              "step_count": step_count,
                              "prediction_type": "only_lambda",
                              "predicted_model": "",
                              "predicted_lambda": estimated_lambda,
                              "predicted_lambda0": estimated_lambda0,
                              "predicted_lambda1": estimated_lambda1,
                              "predicted_p0": ""}
            results = results.append(current_result, ignore_index=True)

            estimated_p0 = get_p0_estimate(walk_type, c_lambdas, walks)
            current_result = {"model_type": walk_type,
                              "c_lambda": c_lambda,
                              "c_lambda0": c_lambda0,
                              "c_lambda1": c_lambda1,
                              "p0": starting_probability,
                              "step_count": step_count,
                              "prediction_type": "only_p0",
                              "predicted_model": "",
                              "predicted_lambda": "",
                              "predicted_lambda0": "",
                              "predicted_lambda1": "",
                              "predicted_p0": estimated_p0}
            results = results.append(current_result, ignore_index=True)

            estimated_params = get_parameters_estimate(walk_type, walks)
            if walk_type == 'success_punished' or walk_type == 'success_rewarded':
                estimated_lambda = estimated_params[1]
                estimated_lambda0 = ""
                estimated_lambda1 = ""
            elif walk_type == 'success_punished_two_lambdas' or walk_type == 'success_rewarded_two_lambdas':
                estimated_lambda = ""
                estimated_lambda0 = estimated_params[1]
                estimated_lambda1 = estimated_params[2]
            else:
                raise Exception(f'Unexpected walk type: {walk_type}')
            current_result = {"model_type": walk_type,
                              "c_lambda": c_lambda,
                              "c_lambda0": c_lambda0,
                              "c_lambda1": c_lambda1,
                              "p0": starting_probability,
                              "step_count": step_count,
                              "prediction_type": "all_parameters",
                              "predicted_model": "",
                              "predicted_lambda": estimated_lambda,
                              "predicted_lambda0": estimated_lambda0,
                              "predicted_lambda1": estimated_lambda1,
                              "predicted_p0": estimated_params[0]}
            results = results.append(current_result, ignore_index=True)

            estimated_params, estimated_model = get_model_estimate(walks)
            if walk_type == 'success_punished' or walk_type == 'success_rewarded':
                estimated_lambda = estimated_params[1]
                estimated_lambda0 = ""
                estimated_lambda1 = ""
            elif walk_type == 'success_punished_two_lambdas' or walk_type == 'success_rewarded_two_lambdas':
                estimated_lambda = ""
                estimated_lambda0 = estimated_params[1]
                estimated_lambda1 = estimated_params[2] if len(estimated_params) == 3 else -10000
            else:
                raise Exception(f'Unexpected walk type: {walk_type}')
            current_result = {"model_type": walk_type,
                              "c_lambda": c_lambda,
                              "c_lambda0": c_lambda0,
                              "c_lambda1": c_lambda1,
                              "p0": starting_probability,
                              "step_count": step_count,
                              "prediction_type": "everything",
                              "predicted_model": estimated_model,
                              "predicted_lambda": estimated_lambda,
                              "predicted_lambda0": estimated_lambda0,
                              "predicted_lambda1": estimated_lambda1,
                              "predicted_p0": estimated_params[0]}
            results = results.append(current_result, ignore_index=True)

            end_time_iter = datetime.now()
            time_curr = end_time_iter - start_time_iter
            time_per_iter = (end_time_iter - start_time_loop) / (i + 1)
            eta = (len(generated_data) - i - 1) * time_per_iter
            logging.info(f"Current iteration: {time_curr}. Per iter {time_per_iter}. ETA: {eta}")

    with open("results.pkl", 'wb') as f:
        pickle.dump([results], f)
    analyze_results(results)


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
    logging.info(f"\nDuration: {(end_time - start_time)}")
