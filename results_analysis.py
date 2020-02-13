import logging
import pickle
from datetime import datetime
from typing import List

import pandas as pd

from config import CONFIDENCE_INTERVAL_SIZE, MODEL_PARAMETERS, PREDICTION_VALUES
from config import C_LAMBDAS, START_PROBABILITIES, STEP_COUNTS, C_LAMBDA_PAIRS, MODEL_TYPES, \
    PREDICTION_TYPES


def parameter_estimate_evaluation(true_parameter_value: float, prediction: float, successes: int) -> int:
    if true_parameter_value * (1 - CONFIDENCE_INTERVAL_SIZE / 2) <= prediction <= true_parameter_value * (
            1 + CONFIDENCE_INTERVAL_SIZE / 2):
        successes = successes + 1
    return successes


def model_estimate_evaluation(true_model_type: str, model_prediction: str, successes: int) -> int:
    if true_model_type == model_prediction:
        successes = successes + 1
    return successes


def check_prediction(prediction: float, model_type: str, prediction_type: str, true_value: float):
    if prediction >= 1 or prediction <= 0:
        logging.error(
            f"Wrong prediction {prediction}. Model {model_type}, Predi_type {prediction_type}, true val {true_value}")


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
                    check_prediction(prediction, model_type, prediction_type, c_lambda)
                    successes = parameter_estimate_evaluation(c_lambda, prediction, successes)
                if prediction_type in ["only_p0", "all_parameters"]:
                    tries = tries + 1
                    prediction = result_of_test.predicted_p0.values[0]
                    check_prediction(prediction, model_type, prediction_type, p0)
                    successes = parameter_estimate_evaluation(p0, prediction, successes)
                if prediction_type in ["everything"]:
                    tries = tries + 1
                    successes = model_estimate_evaluation(model_type, result_of_test.predicted_model.values[0],
                                                          successes)
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
                        check_prediction(prediction, model_type, prediction_type, c_lambda0)
                        successes = parameter_estimate_evaluation(c_lambda0, prediction, successes)

                        tries = tries + 1
                        prediction = result_of_test.predicted_lambda1.values[0]
                        check_prediction(prediction, model_type, prediction_type, c_lambda1)
                        successes = parameter_estimate_evaluation(c_lambda1, prediction, successes)
                    if prediction_type in ["only_p0", "all_parameters"]:
                        tries = tries + 1
                        prediction = result_of_test.predicted_p0.values[0]
                        check_prediction(prediction, model_type, prediction_type, p0)
                        successes = parameter_estimate_evaluation(p0, prediction, successes)
                    if prediction_type in ["everything"]:
                        tries = tries + 1
                        successes = model_estimate_evaluation(model_type, result_of_test.predicted_model.values[0],
                                                              successes)
    logging.info(
        f"Model type: {model_type}, prediction type: {prediction_type}. Successes: {successes}. Tries: {tries}. Success rate: {successes / tries}")


def select_results(results: pd.DataFrame, prediction_type: str, model_type: str, c_lambdas: List[float],
                   step_count: int, p0: float) -> pd.DataFrame:
    results = results[results.prediction_type.isin([prediction_type])]
    results = results[results.model_type.isin([model_type])]
    results = results[results.step_count.isin([step_count])]
    results = results[results.p0.isin([p0])]
    if 'two_lambdas' in model_type:
        results = results[results.c_lambda0.isin([c_lambdas[0]])]
        results = results[results.c_lambda1.isin([c_lambdas[1]])]
    else:
        results = results[results.c_lambda.isin([c_lambdas[0]])]
    return results


def evaluate_lambda_prediction(current_results, model_type, c_lambdas):
    pass


def evaluate_p0_prediction(current_results, model_type, p0):
    pass


def evaluate_model_prediction(current_results, model_type):
    pass


def analyze_prediction_combination(current_results: pd.DataFrame, columns: dict) -> pd.DataFrame:
    input_parameters = current_results.head(1).reset_index(drop=True)
    input_parameters.drop(columns="repetition", inplace=True)
    renaming_columns = {}
    for prediction_value in PREDICTION_VALUES:
        renaming_columns[prediction_value] = f"mean_{prediction_value}"
    input_parameters.rename(columns=renaming_columns, inplace=True)
    input_parameters.drop(columns="mean_predicted_model", inplace=True)
    prediction_type = input_parameters.prediction_type[0]
    model_type = input_parameters.model_type[0]
    c_lambdas = [input_parameters.c_lambda[0], input_parameters.c_lambda0[0], input_parameters.c_lambda1[0]]
    p0 = input_parameters.p0[0]
    result_row = pd.DataFrame(columns=columns)
    result_row = result_row.append(input_parameters, sort=False)
    if prediction_type == "only_lambda" or prediction_type == "all_parameters":
        result_row = evaluate_lambda_prediction(current_results, model_type, c_lambdas)
    if prediction_type == "only_p0" or prediction_type == "all_parameters":
        result_row = evaluate_p0_prediction(current_results, model_type, p0)
    if prediction_type == "everything":
        result_row = evaluate_model_prediction(current_results, model_type)
    return result_row


def analyze_results(results: pd.DataFrame):
    columns = MODEL_PARAMETERS
    columns.extend(["prediction_type", "mean_predicted_lambda", "mean_predicted_lambda0", "mean_predicted_lambda1",
                    "mean_predicted_p0", "median_predicted_lambda", "median_predicted_lambda0", "median_predicted_lambda1",
                    "median_predicted_p0"])
    columns.extend(["predicted_lambda_conf_int_LB", "predicted_lambda_conf_int_UB", "predicted_lambda_percentile_LB",
                    "predicted_lambda_percentile_UP", "predicted_lambda_near_value_LB",
                    "predicted_lambda_near_value_UP"])
    columns.extend(["predicted_lambda0_conf_int_LB", "predicted_lambda0_conf_int_UB", "predicted_lambda0_percentile_LB",
                    "predicted_lambda0_percentile_UP", "predicted_lambda0_near_value_LB",
                    "predicted_lambda0_near_value_UP"])
    columns.extend(["predicted_lambda1_conf_int_LB", "predicted_lambda1_conf_int_UB", "predicted_lambda1_percentile_LB",
                    "predicted_lambda1_percentile_UP", "predicted_lambda1_near_value_LB",
                    "predicted_lambda1_near_value_UP"])
    columns.extend(["predicted_p0_conf_int_LB", "predicted_p0_conf_int_UB", "predicted_p0_percentile_LB",
                    "predicted_p0_percentile_UP", "predicted_p0_near_value_LB", "predicted_p0_near_value_UP"])
    columns.extend(["predicted_lambda_conf_int_mean_success", "predicted_lambda_percentile_mean_success",
                    "predicted_lambda_near_value_mean_success"])
    columns.extend(["predicted_lambda0_conf_int_mean_success", "predicted_lambda0_percentile_mean_success",
                    "predicted_lambda0_near_value_mean_success"])
    columns.extend(["predicted_lambda1_conf_int_mean_success", "predicted_lambda1_percentile_mean_success",
                    "predicted_lambda1_near_value_mean_success"])
    columns.extend(["predicted_p0_conf_int_mean_success", "predicted_p0_percentile_mean_success",
                    "predicted_p0_near_value_mean_success"])    
    columns.extend(["predicted_lambda_conf_int_median_success", "predicted_lambda_percentile_median_success",
                    "predicted_lambda_near_value_median_success"])
    columns.extend(["predicted_lambda0_conf_int_median_success", "predicted_lambda0_percentile_median_success",
                    "predicted_lambda0_near_value_median_success"])
    columns.extend(["predicted_lambda1_conf_int_median_success", "predicted_lambda1_percentile_median_success",
                    "predicted_lambda1_near_value_median_success"])
    columns.extend(["predicted_p0_conf_int_median_success", "predicted_p0_percentile_median_success",
                    "predicted_p0_near_value_median_success"])
    columns.append("predicted_model_success_rate")
    fitting_results = pd.DataFrame(columns=columns)

    for index, c_lambda in enumerate(C_LAMBDAS):
        for p0 in START_PROBABILITIES:
            for step_count in STEP_COUNTS:
                for model_type in MODEL_TYPES:
                    if 'two_lambdas' in model_type:
                        c_lambdas = C_LAMBDA_PAIRS[index]
                    else:
                        c_lambdas = [c_lambda]
                    for prediction_type in PREDICTION_TYPES:
                        current_results = select_results(results, prediction_type, model_type, c_lambdas, step_count,
                                                         p0)
                        fitting_results.append(analyze_prediction_combination(current_results, columns))
        model_types = results['model_type'].unique()
        prediction_types = results['prediction_type'].unique()
        for model_type in model_types:
            for prediction_type in prediction_types:
                if model_type == 'success_punished' or model_type == 'success_rewarded':
                    analyze_result_single_lambda(
                        results[
                            results.model_type.isin([model_type]) & results.prediction_type.isin([prediction_type])],
                        prediction_type, model_type)
                elif model_type == 'success_punished_two_lambdas' or model_type == 'success_rewarded_two_lambdas':
                    analyze_result_multiple_lambda(
                        results[
                            results.model_type.isin([model_type]) & results.prediction_type.isin([prediction_type])],
                        prediction_type, model_type)
                else:
                    raise Exception(f'Unexpected walk type: {model_type}')


def main():
    with open(f"results_total.pkl", 'rb') as f:
        results = pickle.load(f)  # load data

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
    logging.info(f"Duration: {(end_time - start_time)}")
