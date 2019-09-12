# code used to analyze generalized data

import logging
import pickle
from os import listdir
from os.path import isfile, join

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


def main():
    generated_data = [f for f in listdir(DATA_DIRNAME) if isfile(join(DATA_DIRNAME, f))]
    for datafile in generated_data:
        with open(join(DATA_DIRNAME, datafile), 'rb') as f:  # Python 3: open(..., 'rb')
            walks, walk_type, starting_probability, c_lambdas, step_count = pickle.load(f)
            print(walks, walk_type, starting_probability, c_lambdas, step_count)


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
