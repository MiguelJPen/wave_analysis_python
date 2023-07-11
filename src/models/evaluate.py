import torch
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from src.invertedModels.define_loss import loss_function


def evaluate_model(model, test_set, target_range, target_dev, evaluation_graphs):
    loss_fn = loss_function()

    model.eval()
    # Evaluar el modelo
    with torch.no_grad():
        outputs = model(test_set[0].unsqueeze(1))
        test_loss = loss_fn(outputs, test_set[1])

        error = mean_absolute_error(test_set[1].cpu(), outputs.cpu())
        error_by_row = mean_absolute_error(np.transpose(test_set[1].cpu()), np.transpose(outputs.cpu()), multioutput='raw_values')
        error_by_row_sorted = np.sort(error_by_row)

        if evaluation_graphs:
            x_label = [i/10 for i in range(0, 101, 10)]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 8))
            ax1.tick_params(axis='both', labelsize=17)
            ax1.set_title('Método de elementos finitos', size=17)
            sb.heatmap(test_set[1].cpu(), cmap='viridis', cbar=False, yticklabels=60, ax=ax1, vmin=-1.5, vmax=8)
            ax1.set_xlabel('Punto temporal', size=17)
            ax1.set_ylabel('Nº de onda', size=17)
            ax1.xaxis.set_ticks(np.arange(len(x_label)*30, step=30))
            ax1.xaxis.set_ticklabels(x_label)
            ax2.tick_params(axis='both', labelsize=17)
            ax2.set_title('Calculado por la red neuronal', size=17)
            sb.heatmap(outputs.cpu(), cmap='viridis', cbar=False, yticklabels=60, ax=ax2, vmin=-1.5, vmax=8)
            ax2.set_xlabel('Punto temporal', size=17)
            ax2.set_ylabel('Nº de onda', size=17)
            ax2.xaxis.set_ticks(np.arange(len(x_label) * 30, step=30))
            ax2.xaxis.set_ticklabels(x_label)
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(12.5, 7))
            sb.heatmap((test_set[1].cpu() - outputs.cpu()), cmap='viridis', yticklabels=60)
            ax.xaxis.set_ticks(np.arange(len(x_label) * 30, step=30))
            ax.xaxis.set_ticklabels(x_label)
            plt.xlabel('Punto temporal', fontsize=17)
            plt.xticks(fontsize=15)
            plt.ylabel('Nº de onda', fontsize=17)
            plt.yticks(fontsize=15)
            plt.title("Diferencia entre los valores reales y estimados", fontsize=17)
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(12.5, 7))
            sb.heatmap((test_set[1].cpu() - outputs.cpu()), cmap='viridis', yticklabels=60, vmin=-0.5, vmax=0.5)
            ax.xaxis.set_ticks(np.arange(len(x_label) * 30, step=30))
            ax.xaxis.set_ticklabels(x_label)
            plt.xlabel('Punto temporal', fontsize=17)
            plt.xticks(fontsize=15)
            plt.ylabel('Nº de onda', fontsize=17)
            plt.yticks(fontsize=15)
            plt.title("Diferencia entre los valores reales y estimados", fontsize=17)
            plt.show()

        diff_matrix = np.matrix(abs(test_set[1].cpu() - outputs.cpu()))
        mat_range = np.max(np.matrix(test_set[1].cpu())) - np.min(np.matrix(test_set[1].cpu()))

        print("Desviación mayor del 5%: ", "{0:.2%}".format((diff_matrix > mat_range * 0.05).sum()/(diff_matrix < 1000).sum()))

        print("Error absoluto medio en el conjunto de test: ", error * target_dev)
        print("Error relativo: ", "{0:.2%}".format((error * target_dev) / target_range))
        print('Test Loss: {:.4f}'.format(test_loss.item()))

        plt.figure(figsize=(12.5, 5.5))
        plt.title('Menor precisión en la estimación', fontsize=17)
        plt.plot(test_set[1].cpu()[np.array(error_by_row).argmax(), :], label='Valor real')
        plt.plot(outputs.cpu()[np.array(error_by_row).argmax(), :], label='Valor estimado')
        plt.xlabel('Punto temporal', fontsize=17)
        plt.xticks(fontsize=15)
        plt.ylabel('Valor de la onda transformado', fontsize=17)
        plt.yticks(fontsize=15)
        plt.legend(loc="upper right", fontsize=14)
        plt.show()

        plt.figure(figsize=(12.5, 5.5))
        plt.title('Mayor precisión en la estimación', fontsize=17)
        plt.plot(test_set[1].cpu()[np.array(error_by_row).argmin(), :], label='Valor real')
        plt.plot(outputs.cpu()[np.array(error_by_row).argmin(), :], label='Valor estimado')
        plt.xlabel('Punto temporal', fontsize=17)
        plt.xticks(fontsize=15)
        plt.ylabel('Valor de la onda transformado', fontsize=17)
        plt.yticks(fontsize=15)
        plt.legend(loc="upper right", fontsize=14)
        plt.show()

    model.train()
