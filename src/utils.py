import numpy as np


def sigmoide(x, derivada=False):
    """
    Calcula a função de ativação sigmoide para um valor ou array x.

    Args:
        x: o valor ou array onde será aplicada a função
        derivada: retorna o valor da derivada na função em x se True

    Returns:
        O resultado da operação da sigmoide ou da derivada da sigmoide.
    """
    resultado = 1 / (1 + np.exp(-x))
    if derivada:
        return resultado * (1 - resultado)
    else:
        return resultado


def tangente_hiperbolica(x, derivada=False):
    """
    Calcula a função de ativação tangente hiperbólica para um valor ou array x.

    Args:
        x: o valor ou array onde será aplicada a função
        derivada: retorna o valor da derivada na função em x se True

    Returns:
        O resultado da operação da tangente hiperbólica ou de sua derivada.
    """
    resultado = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if derivada:
        return 1 - resultado ** 2
    else:
        return resultado
