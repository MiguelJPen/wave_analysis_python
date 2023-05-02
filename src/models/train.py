import torch
from sklearn.metrics import mean_absolute_error

from src.models.define_loss import loss_function
from src.models.define_optimizer import optimizer


def train_model(model, train_set, validation_set, target_dev):
    # Definir la función de pérdida y el optimizador
    loss_fn = loss_function()
    optim = optimizer(model)

    # Entrenar la red
    for epoch in range(10000):
        outputs = model(train_set[0].unsqueeze(1).cuda())
        loss = loss_fn(outputs, train_set[1])

        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch % 100 == 99:
            model.eval()

            print('Epoch %d loss: %.4f' % (epoch + 1, loss.item() / 100))
            with torch.no_grad():
                outputs = model(validation_set[0].unsqueeze(1))
                val_loss = loss_fn(outputs, validation_set[1])
                error = mean_absolute_error(validation_set[1].cpu(), outputs.cpu())

                print("Error en el conjunto de validación: ", error * target_dev)
                print("Función de pérdida de la validación: ", val_loss.item())
                print("Función de pérdida del entrenamiento: ", loss.item(), end="\n\n")

            model.train()

    print('Finished Training')

    return model