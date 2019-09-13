# support functions


def bernoulli(ising: int) -> int:
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


def ising(bernoulli: int) -> int:
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