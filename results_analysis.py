import logging
import pickle
from datetime import datetime

import pandas as pd

from config import CONFIDENCE_INTERVAL_SIZE


def analyze_result_single_lambda(result: pd.DataFrame, prediction_type: str, model_type: str):
    lambdas = result['c_lambda'].unique()
    p0s = result['p0'].unique()
    step_counts = result['step_count'].unique()
    tries = 0
    successes = 0
    for p0 in p0s:
        if p0 < 0.5:
            continue
        for step_count in step_counts:
            if step_count < 5:
                continue
            for c_lambda in lambdas:
                if c_lambda < 0.5:
                    continue
                result_of_test = result[
                    result.c_lambda.isin([c_lambda]) & result.p0.isin([p0]) & result.step_count.isin([step_count])]
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
                    if model_type in result_of_test.predicted_model.values[0]:
                        successes = successes + 1
    logging.info(
        f"Model type: {model_type}, prediction type: {prediction_type}. Successes: {successes}. Tries: {tries}. Success rate: {successes / tries}")


def analyze_result_multiple_lambda(result, prediction_type, model_type):
    lambdas0 = result['c_lambda0'].unique()
    lambdas1 = result['c_lambda1'].unique()
    p0s = result['p0'].unique()
    step_counts = result['step_count'].unique()
    tries = 0
    successes = 0
    for p0 in p0s:
        if p0 < 0.5:
            continue
        for step_count in step_counts:
            if step_count < 5:
                continue
            for c_lambda0 in lambdas0:
                if c_lambda0 < 0.5:
                    continue
                for c_lambda1 in lambdas1:
                    if c_lambda1 < 0.5:
                        continue
                    result_of_test = result[
                        result.c_lambda0.isin([c_lambda0]) & result.c_lambda1.isin([c_lambda1]) & result.p0.isin(
                            [p0]) & result.step_count.isin([step_count])]
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
                        if result_of_test.predicted_model.values[0] in model_type:
                            successes = successes + 1
    logging.info(
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
    with open("results.pkl", 'rb') as f:
        results = pickle.load(f)  # load data

    analyze_results(results[0])


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
