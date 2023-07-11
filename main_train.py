import torch
import numpy
import matplotlib.pyplot as plt

from src.data import Dataset, data_from_file
from src.models.train import train_model_lr
from src.models.model import build_model

###########################
#  CONTROL FOR THE MODEL  #
###########################
training_dataset = Dataset.lhs_300_10K
num_epochs = 3000
learning_rate = [0.5, 0.1, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
momentum = 0
###########################

((train_set, validation_set, test_set), target_dev), target_range = data_from_file(training_dataset)

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Dispositivo de procesamiento: ", device)

v_loss, val_v_loss = {}, {}
x = numpy.arange(0, num_epochs, 1)
plt.figure(figsize=(12.5, 5.5))

for item in learning_rate:
    model = build_model()
    model = model.to(device)
    (v_loss[item], val_v_loss[item]) = train_model_lr(model, num_epochs, item, momentum, train_set, validation_set)
    plt.plot(x[:len(val_v_loss[item])], val_v_loss[item], label=str(item))

plt.xlabel('Nº de épocas', fontsize=17)
plt.xticks(fontsize=15)
plt.ylabel('Pérdida', fontsize=17)
plt.yticks(fontsize=15)
plt.legend(loc="upper right", fontsize=14)
plt.title("Pérdidad de la red", fontsize=17)
plt.show()
