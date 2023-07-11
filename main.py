import torch

from src.data import Dataset, data_from_file
from src import IO_model
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.model import build_model

###########################
#  CONTROL FOR THE MODEL  #
###########################
training_dataset = Dataset.lhs_300_30K
eval_dataset = Dataset.lhs_300_30K
train_model_bool = False
evaluate_model_bool = True
evaluation_graphs = False
full_testing = False
num_epochs = 3000
learning_rate = 0.1
momentum = 0.99
###########################

if full_testing:
    ((test_set), target_dev), target_range = data_from_file(training_dataset, full=True)
else:
    ((train_set, validation_set, test_set), target_dev), target_range = data_from_file(training_dataset)

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Dispositivo de procesamiento: ", device)

if train_model_bool:
    model = build_model()
    model = model.to(device)
    model = train_model(model, num_epochs, learning_rate, momentum, train_set, validation_set, target_dev)

    IO_model.save_model(model, training_dataset)

if evaluate_model_bool:
    model = IO_model.load_model(eval_dataset)
    model = model.to(device)
    evaluate_model(model, test_set, target_range, target_dev, evaluation_graphs)
