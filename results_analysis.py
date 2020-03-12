import logging
import pickle
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st

from config import CONFIDENCE_INTERVAL_SIZES, MODEL_PARAMETERS, PREDICTION_VALUES, C_LAMBDAS, START_PROBABILITIES, \
    STEP_COUNTS, C_LAMBDA_PAIRS, MODEL_TYPES, PREDICTION_TYPES, REPETITIONS_OF_WALK_S, REPETITIONS_OF_WALK_SERIES, \
    OPTIMIZATION_ALGORITHM


def check_prediction(prediction: pd.Series, model_type: str, prediction_type: str, true_value: float) -> \
        Tuple[pd.Series, int]:
    not_fitted = 0
    if (prediction == 'not_fitted').any():
        not_fitted = sum(prediction == 'not_fitted')
        prediction = prediction[prediction != 'not_fitted']
    if max(prediction) >= 1 or min(prediction) <= 0:
        logging.error(
            f"Wrong prediction {min(prediction), max(prediction)}. Model {model_type}, prediction type "
            f"{prediction_type}, true value {true_value}")
    return prediction, not_fitted


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


def in_interval(datapoint: float, interval: List[float]) -> int:
    return 1 if interval[0] <= datapoint <= interval[1] else 0


def evaluate_point_prediction(result_row: pd.DataFrame, data: pd.Series, true_value: float, name: str,
                              confidence_interval_size: float) -> pd.DataFrame:
    data, not_fitted = check_prediction(data, result_row["model_type"][0], result_row["prediction_type"][0], true_value)
    mean = float(np.mean(data))
    median = float(np.median(data))
    stdev = float(np.std(data))
    conf_int = st.t.interval(1 - confidence_interval_size, len(data) - 1, loc=mean, scale=st.sem(data))
    percentile_int = [np.percentile(data, int(confidence_interval_size / 2 * 100), interpolation='midpoint'),
                      np.percentile(data, int((1 - confidence_interval_size / 2) * 100), interpolation='midpoint')]
    near_int = [max(0.0, true_value - true_value * (confidence_interval_size / 2)),
                min(1.0, true_value + true_value * (confidence_interval_size / 2))]
    conf_int_success = in_interval(true_value, conf_int)
    percentile_success = in_interval(true_value, percentile_int)
    near_mean_success = in_interval(mean, near_int)
    near_median_success = in_interval(median, near_int)
    result_row[f"mean_predicted_{name}"] = mean
    result_row[f"median_predicted_{name}"] = median
    result_row[f"stdev_predicted_{name}"] = stdev
    result_row[f"predicted_{name}_conf_int_LB"] = conf_int[0]
    result_row[f"predicted_{name}_conf_int_UB"] = conf_int[1]
    result_row[f"predicted_{name}_percentile_LB"] = percentile_int[0]
    result_row[f"predicted_{name}_percentile_UP"] = percentile_int[1]
    result_row[f"predicted_{name}_near_value_LB"] = near_int[0]
    result_row[f"predicted_{name}_near_value_UP"] = near_int[1]
    result_row[f"predicted_{name}_conf_int_success"] = conf_int_success
    result_row[f"predicted_{name}_percentile_success"] = percentile_success
    result_row[f"predicted_{name}_near_value_mean_success"] = near_mean_success
    result_row[f"predicted_{name}_near_value_median_success"] = near_median_success
    result_row[f"not_fitted_{name}_count"] = not_fitted
    return result_row


def evaluate_lambda_prediction(result_row: pd.DataFrame, current_results: pd.DataFrame, model_type: str,
                               c_lambdas: List[float], confidence_interval_size: float) -> pd.DataFrame:
    if 'two_lambdas' in model_type:
        data = current_results.predicted_lambda0
        name = "lambda0"
        result_row = evaluate_point_prediction(result_row, data, c_lambdas[1], name, confidence_interval_size)
        data = current_results.predicted_lambda1
        name = "lambda1"
        return evaluate_point_prediction(result_row, data, c_lambdas[2], name, confidence_interval_size)
    else:
        data = current_results.predicted_lambda
        name = "lambda"
        return evaluate_point_prediction(result_row, data, c_lambdas[0], name, confidence_interval_size)


def evaluate_p0_prediction(result_row: pd.DataFrame, current_results: pd.DataFrame, p0: float,
                           confidence_interval_size: float) -> pd.DataFrame:
    data = current_results.predicted_p0
    name = "p0"
    return evaluate_point_prediction(result_row, data, p0, name, confidence_interval_size)


def evaluate_model_prediction(result_row: pd.DataFrame, current_results: pd.DataFrame, model_type: str) -> pd.DataFrame:
    data = current_results.predicted_model
    not_fitted = 0
    if (data == 'not_fitted').any():
        not_fitted = sum(data == 'not_fitted')
    result_row["predicted_model_success_rate"] = len(data[data == model_type]) / len(data)
    result_row["mean_predicted_lambda"] = ""
    result_row["mean_predicted_lambda0"] = ""
    result_row["mean_predicted_lambda1"] = ""
    result_row["mean_predicted_p0"] = ""
    result_row["not_fitted_model_count"] = not_fitted
    return result_row


def analyze_prediction_combination(current_results: pd.DataFrame, columns: dict,
                                   confidence_interval_size: float) -> pd.DataFrame:
    # gather input parameters
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

    # prepare result dataframe - actually only a row
    result_row = pd.DataFrame(columns=columns)
    result_row = result_row.append(input_parameters, sort=False)

    # evaluate fitting
    if prediction_type == "only_lambda" or prediction_type == "all_parameters":
        result_row = evaluate_lambda_prediction(result_row, current_results, model_type, c_lambdas,
                                                confidence_interval_size)
    if prediction_type == "only_p0" or prediction_type == "all_parameters":
        result_row = evaluate_p0_prediction(result_row, current_results, p0, confidence_interval_size)
    if prediction_type == "everything":
        result_row = evaluate_model_prediction(result_row, current_results, model_type)
    return result_row


def analyze_results(results: pd.DataFrame, repetitions_of_walk: int):
    columns = MODEL_PARAMETERS
    columns.extend(["prediction_type", "mean_predicted_lambda", "mean_predicted_lambda0", "mean_predicted_lambda1",
                    "mean_predicted_p0", "stdev_predicted_lambda", "stdev_predicted_lambda0", "stdev_predicted_lambda1",
                    "stdev_predicted_p0", "median_predicted_lambda", "median_predicted_lambda0",
                    "median_predicted_lambda1", "median_predicted_p0"])
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
    columns.extend(["predicted_lambda_conf_int_success", "predicted_lambda_percentile_success",
                    "predicted_lambda_near_value_mean_success", "predicted_lambda_near_value_median_success"])
    columns.extend(["predicted_lambda0_conf_int_success", "predicted_lambda0_percentile_success",
                    "predicted_lambda0_near_value_mean_success", "predicted_lambda0_near_value_median_success"])
    columns.extend(["predicted_lambda1_conf_int_success", "predicted_lambda1_percentile_success",
                    "predicted_lambda1_near_value_mean_success", "predicted_lambda1_near_value_median_success"])
    columns.extend(["predicted_p0_conf_int_success", "predicted_p0_percentile_success",
                    "predicted_p0_near_value_mean_success", "predicted_p0_near_value_median_success"])
    columns.append("predicted_model_success_rate")
    columns.extend(["not_fitted_lambda_count", "not_fitted_lambda0_count", "not_fitted_lambda1_count",
                    "not_fitted_p0_count", "not_fitted_model_count"])
    fitting_results = pd.DataFrame(columns=columns)

    for confidence_interval_size in CONFIDENCE_INTERVAL_SIZES:
        for index, c_lambda in enumerate(C_LAMBDAS):
            for p0 in START_PROBABILITIES:
                for step_count in STEP_COUNTS:
                    for model_type in MODEL_TYPES:
                        if 'two_lambdas' in model_type:
                            c_lambdas = C_LAMBDA_PAIRS[index]
                        else:
                            c_lambdas = [c_lambda]
                        for prediction_type in PREDICTION_TYPES:
                            current_results = select_results(results, prediction_type, model_type, c_lambdas,
                                                             step_count,
                                                             p0)
                            fitting_results = fitting_results.append(
                                analyze_prediction_combination(current_results, columns, confidence_interval_size))
        fitting_results.to_excel(
            f"fitting_evaluation_interval_size_{confidence_interval_size}_K{repetitions_of_walk}_"
            f"N{REPETITIONS_OF_WALK_SERIES}_{OPTIMIZATION_ALGORITHM}.xlsx")


def main():
    for repetitions_of_walk in REPETITIONS_OF_WALK_S:
        with open(f"results_{OPTIMIZATION_ALGORITHM}_K{repetitions_of_walk}_N{REPETITIONS_OF_WALK_SERIES}.pkl",
                  'rb') as f:
            results = pickle.load(f)  # load data
            if isinstance(results, list):
                results = results[0]
            analyze_results(results, repetitions_of_walk)


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
