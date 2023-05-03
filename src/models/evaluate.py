import torch
from sklearn.metrics import mean_absolute_error
import seaborn as sb
import matplotlib.pyplot as plt

from src.invertedModels.define_loss import loss_function


def evaluate_model(model, test_set, target_range, target_dev):
    loss_fn = loss_function()

    model.eval()
    # Evaluar el modelo
    with torch.no_grad():
        outputs = model(test_set[0].unsqueeze(1))
        test_loss = loss_fn(outputs, test_set[1])

        error = mean_absolute_error(test_set[1].cpu(), outputs.cpu())

        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #sb.heatmap(test_set[1].cpu(), cmap='viridis', xticklabels=False, yticklabels=False, ax=ax1)
        #sb.heatmap(outputs.cpu(), cmap='viridis', xticklabels=False, yticklabels=False, ax=ax2)
        sb.heatmap((test_set[1].cpu() - outputs.cpu()), cmap='viridis', xticklabels=False, yticklabels=False)

        plt.show()

        print("Error en el conjunto de test: ", error * target_dev)
        print("Error relativo: ", "{0:.2%}".format((error * target_dev) / target_range))
        print('Test Loss: {:.4f}'.format(test_loss.item()))

    model.train()
