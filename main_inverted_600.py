import torch

from src.data import Dataset, data_from_file
from src import IO_model
from src.invertedModels.train import train_model
from src.invertedModels.evaluate import evaluate_model
from src.invertedModels.in_ch_600 import build_600in_net

###########################
#  CONTROL FOR THE MODEL  #
###########################
training_dataset = Dataset.lhs_600_t20_50K
eval_dataset = Dataset.lhs_600_t20_50K
train_model_bool = True
evaluate_model_bool = True
num_epochs = 3000
learning_rate = 0.001
momentum = 0.99
###########################

((train_set, validation_set, test_set), target_dev), target_range = data_from_file(training_dataset, inverted=True)

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Dispositivo de procesamiento: ", device)

if train_model_bool:
    model = build_600in_net()
    model = model.to(device)
    model = train_model(model, num_epochs, learning_rate, momentum, train_set, validation_set, target_dev)

    IO_model.save_model(model, training_dataset, inverted=True, long=True)

if evaluate_model_bool:
    model = IO_model.load_model(eval_dataset, inverted=True, long=True)
    model = model.to(device)
    evaluate_model(model, test_set, target_range, target_dev)
