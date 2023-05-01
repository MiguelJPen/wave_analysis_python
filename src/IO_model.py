import torch

PATH = 'trainedModel/model_scripted.pt'


def save_model(model):
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(PATH)  # Save


def load_model():
    model = torch.jit.load(PATH)
    model.eval()
