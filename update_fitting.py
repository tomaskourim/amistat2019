# code used to re-fit non-fitted models

import logging
import os
import pickle
from datetime import datetime
from decimal import *
from typing import List

import pandas as pd

from config import DATA_DIRNAME, OPTIMIZATION_ALGORITHM, REPETITIONS_OF_WALK_S, \
    REPETITIONS_OF_WALK_SERIES, ERROR_VALUE, DECIMAL_PRECISION
from data_generation import generate_and_save_walks
from model_fitting import get_basic_result, estimate_current_walks_model, estimate_current_walks_all, \
    estimate_current_walks_lambda, estimate_current_walks_p0
from results_analysis import select_results


def get_walks(model_type: str, p0: float, c_lambdas: List[float], step_count: int, repetition: int,
              repetitions_of_walk: int) -> List[List[int]]:
    filename = f"{DATA_DIRNAME}/K{repetitions_of_walk}/" \
               f"{model_type}__start{p0}__lambdas{c_lambdas}__steps{step_count}__repetition{repetition}.pkl"
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            walks = pickle.load(f)[0]  # load data
    else:
        walks = generate_and_save_walks(model_type, p0, c_lambdas, step_count, repetitions_of_walk, repetition)
    return walks


def fix_fitting(result_row: pd.Series, repetitions_of_walk: int) -> pd.Series:
    p0 = result_row.p0
    model_type = result_row.model_type
    c_lambdas = [result_row.c_lambda] if 'two_lambdas' not in model_type else [result_row.c_lambda0,
                                                                               result_row.c_lambda1]
    step_count = result_row.step_count
    repetition = result_row.repetition
    walks = get_walks(model_type, p0, c_lambdas, step_count, repetition, repetitions_of_walk)
    basic_result = get_basic_result(model_type, result_row.c_lambda, result_row.c_lambda0, result_row.c_lambda1, p0,
                                    step_count, repetition)
    prediction_type = result_row.prediction_type
    if prediction_type == 'everything':
        current_result = estimate_current_walks_model(walks, model_type, basic_result)
    elif prediction_type == 'all_parameters':
        current_result = estimate_current_walks_all(walks, model_type, basic_result)
    elif prediction_type == 'only_p0':
        current_result = estimate_current_walks_p0(walks, model_type, c_lambdas, basic_result)
    elif prediction_type == 'only_lambda':
        current_result = estimate_current_walks_lambda(walks, model_type, p0, basic_result)
    else:
        logging.exception(f"Wrong model type {model_type}")
        raise Exception(f"Wrong model type {model_type}")
    return pd.Series(current_result)


def update_results_selected(results: pd.DataFrame, repetitions_of_walk: int):
    refitting_count = 0
    with open(f"to-refit_K{repetitions_of_walk}.pkl", 'rb') as f:
        refit = pickle.load(f)
    # refit = []
    # refit.append([["success_punished"], [0.5], [""], [""], [0.5], [5], ["only_lambda"]])
    # refit.append([["success_rewarded"], [0.5], [""], [""], [0.5], [5], ["only_lambda"]])
    # refit.append([["success_punished_two_lambdas"], [""], [0.5], [0.8], [0.5], [5], ["only_lambda"]])
    # refit.append([["success_rewarded_two_lambdas"], [""], [0.5], [0.8], [0.5], [5], ["only_lambda"]])
    # refit.append([["success_punished"], [0.5], [""], [""], [0.5], [100], ["only_lambda"]])
    # refit.append([["success_rewarded"], [0.5], [""], [""], [0.5], [100], ["only_lambda"]])
    # refit.append([["success_punished_two_lambdas"], [""], [0.5], [0.8], [0.5], [100], ["only_lambda"]])
    # refit.append([["success_rewarded_two_lambdas"], [""], [0.5], [0.8], [0.5], [100], ["only_lambda"]])
    while len(refit) > 0:
        refitting = refit[0]

        prediction_type = refitting[6][0]
        model_type = refitting[0][0]
        if 'two_lambdas' in model_type:
            c_lambdas = [refitting[2][0], refitting[3][0]]
        else:
            c_lambdas = [refitting[1][0]]
        step_count = refitting[5][0]
        p0 = refitting[4][0]

        current_results = select_results(results, prediction_type, model_type, c_lambdas, step_count, p0)

        for index, result_row in current_results.iterrows():
            refitting_count = refitting_count + 1
            new_results = fix_fitting(result_row, repetitions_of_walk)
            results.loc[index] = new_results
            with open(f"results_{OPTIMIZATION_ALGORITHM}_K{repetitions_of_walk}_N{REPETITIONS_OF_WALK_SERIES}.pkl",
                      'wb') as f:
                pickle.dump([results], f)
            print(f"Success = {refitting_count}, remaining = {len(refit) * 100}")
        refit.pop(0)
        with open(f"to-refit_K{repetitions_of_walk}.pkl", 'wb') as f:
            pickle.dump(refit, f)


def update_results_not_fitted(results: pd.DataFrame, repetitions_of_walk: int):
    success_count = 0
    error_count = 0
    with open(f"to-refit_K{repetitions_of_walk}.pkl", 'rb') as f:
        refit = pickle.load(f)

    for index, result_row in results.iterrows():
        if (result_row.prediction_type == 'everything' and result_row.predicted_model == ERROR_VALUE) \
                or \
                (result_row.prediction_type == 'only_lambda' and (
                        result_row.predicted_lambda == ERROR_VALUE or result_row.predicted_lambda0 == ERROR_VALUE or
                        result_row.predicted_lambda1 == ERROR_VALUE)) \
                or \
                (result_row.prediction_type == 'only_p0' and result_row.predicted_p0 == ERROR_VALUE) \
                or \
                (result_row.prediction_type == 'all_parameters' and (
                        result_row.predicted_lambda == ERROR_VALUE or result_row.predicted_lambda0 == ERROR_VALUE or
                        result_row.predicted_lambda1 == ERROR_VALUE or result_row.predicted_p0 == ERROR_VALUE)):
            new_results = fix_fitting(result_row, repetitions_of_walk)
            if (new_results == ERROR_VALUE).any():
                error_count = error_count + 1
            else:
                success_count = success_count + 1
            results.loc[index] = new_results
            with open(f"results_{OPTIMIZATION_ALGORITHM}_K{repetitions_of_walk}_N{REPETITIONS_OF_WALK_SERIES}.pkl",
                      'wb') as f:
                pickle.dump([results], f)
            print(f"Success = {success_count}, Error = {error_count}")


def main():
    getcontext().prec = DECIMAL_PRECISION
    for repetitions_of_walk in REPETITIONS_OF_WALK_S:
        with open(f"results_{OPTIMIZATION_ALGORITHM}_K{repetitions_of_walk}_N{REPETITIONS_OF_WALK_SERIES}.pkl",
                  'rb') as f:
            results = pickle.load(f)  # load data
        if isinstance(results, list):
            results = results[0]
        update_results_selected(results, repetitions_of_walk)


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
