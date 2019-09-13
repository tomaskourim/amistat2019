# support functions
from typing import List


def ising2bernoulli(ising: int) -> int:
    """
    Transfers variable form Ising random to Bernoulli random; see
    https://en.wikipedia.org/wiki/Bernoulli_distribution
    https://en.wikipedia.org/wiki/Ising_model

    :param ising: int
    :return:
    """
    if ising == 1:
        return 1
    elif ising == -1:
        return 0
    else:
        raise Exception(f'Unexpected value of Ising distribution: {ising}')


def bernoulli2ising(bernoulli: int) -> int:
    """
    Transfers variable form Bernoulli random to Ising random; see
    https://en.wikipedia.org/wiki/Bernoulli_distribution
    https://en.wikipedia.org/wiki/Ising_model

    :param bernoulli: int
    :return:
    """
    if bernoulli == 1:
        return 1
    elif bernoulli == 0:
        return -1
    else:
        raise Exception(f'Unexpected value of Bernoulli distribution: {bernoulli}')


def get_current_probability(c_lambdas: List[float], last_probability: float, step: int, walk_type: str) -> float:
    """
    Computes the transition probability for the next step according to the respective definition as in the paper.
    :param c_lambdas:
    :param last_probability:
    :param step:
    :param walk_type:
    :return:
    """
    if walk_type == 'success_punished':
        return c_lambdas[0] * last_probability + 0.5 * (1 - c_lambdas[0]) * (1 - step)
    elif walk_type == 'success_rewarded':
        return c_lambdas[0] * last_probability + 0.5 * (1 - c_lambdas[0]) * (1 + step)
    elif walk_type == 'success_punished_two_lambdas':
        return 0.5 * ((1 + step) * c_lambdas[0] * last_probability + (1 - step) * (
                1 - c_lambdas[1] * (1 - last_probability)))
    elif walk_type == 'success_rewarded_two_lambdas':
        return 0.5 * ((1 - step) * c_lambdas[0] * last_probability + (1 + step) * (
                1 - c_lambdas[1] * (1 - last_probability)))
    else:
        raise Exception(f'Unexpected walk type: {walk_type}')
