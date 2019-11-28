# support functions
from typing import List


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
    :param step: as Ising variable
    :param walk_type:
    :return:
    """
    if step == '':  # at the beginning of the walk just return p0
        return last_probability
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


class CompleteWalk:
    def __init__(self, probabilities, steps, development):
        self.probabilities = probabilities
        self.steps = steps
        self.development = development
