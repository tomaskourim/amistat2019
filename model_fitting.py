# code used to analyze generalized data

import logging
import pickle
from os import listdir
from os.path import isfile, join
from typing import List

import numpy as np
import scipy.optimize as opt

from common import bernoulli
from config import DATA_DIRNAME


# iterate over all generated cases
# load data
# try different possible models
# find the best suitable model & parameter values
# compare with reality
# p0 known, get lambda
# lambda known, get p)
# first estimate p0, then get lambda
# get lambda and p0


def negative_log_likelihood(c_lambda: float, walk_type: str, starting_probability: float,
                            walks: List[List[int]]) -> float:
    log_likelihood = 0
    for walk in walks:
        current_probability = starting_probability
        for i in range(2, len(walk)):
            current_probability = c_lambda * current_probability + 0.5 * (1 - c_lambda) * (1 - walk[i - 1])
            result = bernoulli(walk[i])
            log_likelihood = log_likelihood + np.log(
                current_probability * result + (1 - current_probability) * (1 - result))
    return -log_likelihood


def estimate_lambda(walk_type: str, starting_probability: float, walks: List[List[int]]):
    opt_result = opt.minimize_scalar(negative_log_likelihood, bounds=(0, 1), method='bounded',
                                     args=(walk_type, starting_probability, walks))
    if opt_result.success:
        logging.info("Fitted successfully.")
        return opt_result.x
    else:
        return None
    pass


def main():
    generated_data = [f for f in listdir(DATA_DIRNAME) if isfile(join(DATA_DIRNAME, f))]
    i = 0
    for datafile in generated_data:
        with open(join(DATA_DIRNAME, datafile), 'rb') as f:  # Python 3: open(..., 'rb')
            walks, walk_type, starting_probability, c_lambdas, step_count = pickle.load(f)
            if walk_type == 'success_punished':
                estimated_lambda = estimate_lambda(walk_type, starting_probability, walks)
                if abs(c_lambdas[0] - estimated_lambda > 0.01):
                    i = i + 1
                    print(i, starting_probability, step_count, estimated_lambda, c_lambdas)
            elif walk_type == 'success_rewarded':
                continue
            elif walk_type == 'success_punished_two_lambdas':
                continue
            elif walk_type == 'success_rewarded_two_lambdas':
                continue
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
