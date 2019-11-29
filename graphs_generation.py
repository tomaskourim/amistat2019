# used to generate useful graphics
import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from common import expected_p_t_array, var_p_t_array
from config import WALK_TYPES, REPETITIONS, \
    C_LAMBDAS_TESTING, START_PROBABILITIES_TESTING, STEP_COUNTS_TESTING, C_LAMBDA_PAIRS_TESTING
from data_generation import generate_random_walks, list_walks2list_lists


def main():
    plt_rows = 1
    plt_columns = 3
    mean_styles = ['g.', 'r.', 'b.']
    var_styles = ['g-.', 'r-.', 'b-.']
    expected_styles = ['g-', 'r-', 'b-']
    for step_count in STEP_COUNTS_TESTING:
        for walk_type in WALK_TYPES:
            if 'two_lambdas' in walk_type:
                two_lambda = True
            else:
                two_lambda = False
            for p_index, starting_probability in enumerate(START_PROBABILITIES_TESTING):
                plt.subplot(plt_rows, plt_columns, p_index + 1)
                plt.axis([0, step_count, -0.1, 1])
                plt.title(r'$p_{0}=%.2f$' % starting_probability)
                plt.xlabel('steps')
                # plt.ylabel('Var(P(t)) / EP(t)')
                for index, c_lambda in enumerate(C_LAMBDAS_TESTING):
                    if two_lambda:
                        c_lambdas = C_LAMBDA_PAIRS_TESTING[index]
                        label = r'$\bar{\lambda}=[%.2f,%.2f]$' % (c_lambdas[0], c_lambdas[1])
                    else:
                        c_lambdas = [c_lambda]
                        label = r'$\lambda=%.2f$' % c_lambda
                    walks = generate_random_walks(walk_type, starting_probability, c_lambdas, step_count, REPETITIONS)
                    probabilities, steps, developments = list_walks2list_lists(walks)

                    mean_probability = np.mean(probabilities, axis=0)
                    variance_probability = np.var(probabilities, axis=0)
                    plt.plot(mean_probability, mean_styles[index], label=label)
                    plt.plot(variance_probability, var_styles[index])
                    if not two_lambda:
                        plt.plot(expected_p_t_array(step_count, starting_probability, c_lambda, walk_type),
                                 expected_styles[index], linewidth=0.7)
                        plt.plot(var_p_t_array(step_count, starting_probability, c_lambda, walk_type),
                                 expected_styles[index], linewidth=0.7)
                    plt.legend(loc='best', fontsize='medium')

            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            fig.show()
            fig.savefig(f'ept_{REPETITIONS}_walks_{step_count}_steps_type_{walk_type}.pdf', dpi=100)


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
