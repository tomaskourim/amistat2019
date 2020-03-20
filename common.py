# support functions
from decimal import *
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


def get_current_probability(c_lambdas: List[float], last_probability: Decimal, step: int, walk_type: str) -> Decimal:
    """
    Computes the transition probability for the next step according to the respective definition as in the paper.
    :param c_lambdas:
    :param last_probability:
    :param step: as Ising variable
    :param walk_type:
    :return:
    """
    if step == '' or step == 0:  # at the beginning of the walk just return p0
        return last_probability
    if walk_type == 'success_punished':
        return Decimal(c_lambdas[0]) * last_probability + Decimal(0.5 * (1 - c_lambdas[0]) * (1 - step))
    elif walk_type == 'success_rewarded':
        return Decimal(c_lambdas[0]) * last_probability + Decimal(0.5 * (1 - c_lambdas[0]) * (1 + step))
    elif walk_type == 'success_punished_two_lambdas':
        return Decimal(0.5) * (Decimal((1 + step) * c_lambdas[0]) * last_probability + (1 - step) * (
                Decimal(1) - Decimal(c_lambdas[1]) * (Decimal(1) - last_probability)))
    elif walk_type == 'success_rewarded_two_lambdas':
        return Decimal(0.5) * (Decimal((1 - step) * c_lambdas[0]) * last_probability + Decimal((1 + step) * (
                Decimal(1) - Decimal(c_lambdas[1]) * (Decimal(1) - last_probability))))
    else:
        raise Exception(f'Unexpected walk type: {walk_type}')


class CompleteWalk:
    def __init__(self, probabilities, steps, development):
        self.probabilities = probabilities
        self.steps = steps
        self.development = development


def expected_p_t_array(step_count: int, p0: float, c_lambda: float, model_type: str) -> List[float]:
    e_array = []
    for step in range(0, step_count + 1):
        e_array.append(expected_p_t(step, p0, c_lambda, model_type))
    return e_array


def expected_p_t(step: int, p0: float, c_lambda: float, model_type: str) -> float:
    """
    Computes expected value of transition probability according to theoretical results.
    :param step:
    :param p0:
    :param c_lambda:
    :param model_type:
    :return:
    """
    if model_type == 'success_punished':
        e = (2 * c_lambda - 1) ** step * p0 + (1 - (2 * c_lambda - 1) ** step) / 2 if c_lambda != 0.5 else 0.5
    elif model_type == 'success_rewarded':
        e = p0
    elif model_type == 'success_punished_two_lambdas':
        e = 0
    elif model_type == 'success_rewarded_two_lambdas':
        e = 0
    else:
        raise Exception(f'Unexpected walk type: {model_type}')
    return e


def support_k(i: int, p0: float, c_lambda: float):
    return (expected_p_t(i, p0, c_lambda, "success_punished") * (-3 * c_lambda ** 2 + 4 * c_lambda - 1) + (
            1 - c_lambda) ** 2)


def expected_p_t_squared_support_sum(step: int, p0: float, c_lambda: float, model_type: str) -> float:
    e = 0
    for i in range(1, step + 1):
        if model_type == 'success_punished':
            summand = support_k(i - 1, p0, c_lambda) * (3 * c_lambda ** 2 - 2 * c_lambda) ** (step - i)
        elif model_type == 'success_rewarded':
            summand = (2 * c_lambda - c_lambda ** 2) ** (step - i)
        elif model_type == 'success_punished_two_lambdas':
            summand = 0
        elif model_type == 'success_rewarded_two_lambdas':
            summand = 0
        else:
            raise Exception(f'Unexpected walk type: {model_type}')
        e = e + summand
    return e


def expected_p_t_squared(step: int, p0: float, c_lambda: float, model_type: str) -> float:
    """
    Support function to get the variance
    :param step:
    :param p0:
    :param c_lambda:
    :param model_type:
    :return:
    """

    support_sum = expected_p_t_squared_support_sum(step, p0, c_lambda, model_type)

    if model_type == 'success_punished':
        e = (3 * c_lambda ** 2 - 2 * c_lambda) ** step + support_sum if c_lambda != 2 / 3 else 0
    elif model_type == 'success_rewarded':
        e = (2 * c_lambda - c_lambda ** 2) ** step * p0 ** 2 + p0 * (1 - c_lambda) ** 2 * support_sum
    elif model_type == 'success_punished_two_lambdas':
        e = 0
    elif model_type == 'success_rewarded_two_lambdas':
        e = 0
    else:
        raise Exception(f'Unexpected walk type: {model_type}')
    return e


def var_p_t_array(step_count: int, p0: float, c_lambda: float, model_type: str) -> List[float]:
    var_array = [p0 * (1 - p0)]  # VarP(0)
    for step in range(1, step_count + 1):
        ep2 = expected_p_t_squared(step, p0, c_lambda, model_type)
        ep = expected_p_t(step, p0, c_lambda, model_type)
        var_array.append(ep2 - ep ** 2)
    return var_array


def list_walks2list_lists(walks):
    """
    Takes a list of complete walks on the input and returns it as a list of lists.
    :param walks:
    :return:
    """
    walks_steps = []
    walks_developments = []
    walks_probabilities = []
    for walk in walks:
        walks_steps.append(walk.steps)
        walks_developments.append(walk.development)
        walks_probabilities.append(walk.probabilities)
    return [walks_probabilities, walks_steps, walks_developments]
