import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.invertedModels.define_loss import loss_function


def evaluate_model(model, test_set, target_range, target_dev):
    loss_fn = loss_function()

    model.eval()
    # Evaluar el modelo
    with torch.no_grad():
        outputs = model(test_set[0].unsqueeze(1))
        test_loss = loss_fn(outputs, test_set[1])

        errors = [mean_absolute_error(x, y) for x, y in
                  zip(torch.transpose(test_set[1].cpu(), 0, 1), torch.transpose(outputs.cpu(), 0, 1))]

        results_0 = np.array([(torch.transpose(test_set[1].cpu(), 0, 1)[0]).tolist(), (torch.transpose(outputs.cpu(), 0, 1)[0]).tolist()]).T * 1.15470053 + 5
        results_1 = np.array([(torch.transpose(test_set[1].cpu(), 0, 1)[1]).tolist(), (torch.transpose(outputs.cpu(), 0, 1)[1]).tolist()]).T * 1.38564068 + 2.6
        results_2 = np.array([(torch.transpose(test_set[1].cpu(), 0, 1)[2]).tolist(), (torch.transpose(outputs.cpu(), 0, 1)[2]).tolist()]).T * 2.30940101 + 5
        results_0 = np.sort(results_0, axis=0)
        results_1 = np.sort(results_1, axis=0)
        results_2 = np.sort(results_2, axis=0)

        plt.figure(figsize=(12.5, 6))
        plt.plot(results_0)
        plt.ylabel('Valor para $x_0$', fontsize=17)
        plt.yticks(fontsize=15)
        plt.xlabel('Nº de onda', fontsize=17)
        plt.xticks(fontsize=15)
        plt.title("Diferencia entre los valores reales y estimados para $x_0$", fontsize=17)
        plt.legend(['Valores reales', 'Valores estimados'], loc="upper right", fontsize=14)
        plt.show()

        plt.figure(figsize=(12.5, 6))
        plt.plot(results_1)
        plt.ylabel('Valor para $l_0$', fontsize=17)
        plt.yticks(fontsize=15)
        plt.xlabel('Nº de onda', fontsize=17)
        plt.xticks(fontsize=15)
        plt.title("Diferencia entre los valores reales y estimados para $l_0$", fontsize=17)
        plt.legend(['Valores reales', 'Valores estimados'], loc="upper right", fontsize=14)
        plt.show()

        plt.figure(figsize=(12.5, 6))
        plt.plot(results_2)
        plt.ylabel('Valor para $c_0$', fontsize=17)
        plt.yticks(fontsize=15)
        plt.xlabel('Nº de onda', fontsize=17)
        plt.xticks(fontsize=15)
        plt.title("Diferencia entre los valores reales y estimados para $c_0$", fontsize=17)
        plt.legend(['Valores reales', 'Valores estimados'], loc="upper right", fontsize=14)
        plt.show()

        print("Error en el conjunto de test: ", np.array(errors) * target_dev)
        print("Error relativo por parámetro: ", ["{0:.2%}".format(ele) for ele in ((np.array(errors) * target_dev) / target_range)])
        print('Test Loss: {:.4f}'.format(test_loss.item()))

    model.train()
