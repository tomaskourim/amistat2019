import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from config import MODEL_PARAMETERS, PREDICTION_VALUES


def main():
    result_files = ['results_archive/results_Nelder-Mead_constrained.pkl', 'results_archive/results_bettor1.pkl',
                    'results_archive/results_amistat1.pkl', 'results_archive/results_amistat2.pkl']
    columns = MODEL_PARAMETERS
    columns.append("prediction_type")
    columns.extend(PREDICTION_VALUES)
    columns.append("repetition")
    results_total = pd.DataFrame(columns=columns)
    for file in result_files:
        with open(file, 'rb') as f:
            results = pickle.load(f)  # load data
        current_repetitions = results_total.repetition.max()
        results[0].repetition += (current_repetitions + 1) if not np.isnan(current_repetitions) else 0
        results_total = results_total.append(results[0])
    with open(f"results_total.pkl", 'wb') as f:
        pickle.dump(results_total, f)


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
