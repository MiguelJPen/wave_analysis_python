import torch.optim as optim


def optimizer(model):
    """
    El optimizador optim.SGD se utiliza para aplicar el algoritmo de optimización del descenso de gradiente
    estocástico (SGD) para actualizar los pesos de la red durante el entrenamiento.
    """
    learning_rate = 0.03 # 0.003

    return optim.SGD(model.parameters(), momentum=0.75, lr=learning_rate)
