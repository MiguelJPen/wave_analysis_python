import torch.optim as optim


def optimizer(model, learning_rate, momentum):
    """
    El optimizador optim.SGD se utiliza para aplicar el algoritmo de optimización del descenso de gradiente
    estocástico (SGD) para actualizar los pesos de la red durante el entrenamiento.
    """

    return optim.SGD(model.parameters(), momentum=momentum, lr=learning_rate)
