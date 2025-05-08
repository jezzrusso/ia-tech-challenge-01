import numpy as np

def get_next_challenge_age(age, available_ages):
    """Retorna a próxima idade de desafio a partir de uma lista de idades disponíveis."""
    next_ages = available_ages[available_ages >= age]
    if len(next_ages) > 0:
        return next_ages.min()
    return np.nan  # Caso não haja idade maior (ex.: idade > máxima disponível)