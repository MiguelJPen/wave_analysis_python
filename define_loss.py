import torch.nn as nn

def loss_function():
    """
    La función de pérdida nn.CrossEntropyLoss() se utiliza para clasificación multiclase.
    La función de entropía cruzada es adecuada para problemas de clasificación en los que el objetivo es predecir
    la probabilidad de que cada clase sea la correcta. La función de pérdida compara la distribución de
    probabilidad predicha por la red con la distribución de probabilidad correcta y penaliza la predicción incorrecta.
    """
    loss_fn = nn.MSELoss()

    return loss_fn
