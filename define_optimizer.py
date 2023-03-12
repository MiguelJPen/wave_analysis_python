import torch.optim as optim


def optimizer(model):
    """
    El optimizador optim.SGD se utiliza para aplicar el algoritmo de optimización del descenso de gradiente
    estocástico (SGD) para actualizar los pesos de la red durante el entrenamiento.
    """
    leaning_rate = 1

    return optim.SGD(model.parameters(), momentum=0.9, lr=leaning_rate)
