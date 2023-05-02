import torch

PATH = 'trainedModel/'


def save_model(model, dataset, inverted=False):
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(PATH + dataset[2] + ('inverted' if inverted else '') + '.pt')  # Save


def load_model(dataset, inverted=False):
    model = torch.jit.load(PATH + dataset[2] + ('inverted' if inverted else '') + '.pt')
    model.eval()

    return model
