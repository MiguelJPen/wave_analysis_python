import numpy as np

import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess.data import get_data, Dataset
from define_loss import loss_function
from define_optimizer import optimizer
import define_model
import IO_model
import test_model

torch.cuda.empty_cache()

(train_dt, val_dt, test_dt, train_tg, val_tg, test_tg), target_dev = get_data(Dataset.lhs_10K_100)

net = define_model.create_net_100()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Dispositivo de procesamiento:", device)

net = net.to(device)
train_dt, train_tg = train_dt.cuda(), train_tg.cuda()
val_dt, val_tg = val_dt.cuda(), val_tg.cuda()
test_dt, test_tg = test_dt.cuda(), test_tg.cuda()

# Definir la función de pérdida y el optimizador
criterion = loss_function()
optim = optimizer(net)

# Entrenar la red
for epoch in range(80):
    running_loss = 0.0

    outputs = net(train_dt.unsqueeze(1).cuda())

    loss = criterion(outputs, train_tg)
    loss.backward()
    optim.step()
    running_loss += loss.item()
    optim.zero_grad()

    if epoch % 10 == 9:
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / 100))

        net.eval()
        with torch.no_grad():
            outputs = net(val_dt.unsqueeze(1))
            test_loss = loss_function()(outputs, val_tg)
            errors = [mean_absolute_error(x, y) for x, y in
                      zip(torch.transpose(val_tg.cpu(), 0, 1), torch.transpose(outputs.cpu(), 0, 1))]
            print("Error en el conjunto de validación: ", np.array(errors) * target_dev)

        net.train()

print('Finished Training')

# IO_model.save_model(net)

net.eval()
# Evaluar el modelo
with torch.no_grad():
    # Obtener las predicciones para los datos de prueba
    outputs = net(test_dt.unsqueeze(1))
    # Calcular la pérdida para los datos de prueba
    test_loss = loss_function()(outputs, test_tg)

    errors = [mean_absolute_error(x, y) for x, y in zip(torch.transpose(test_tg.cpu(), 0, 1), torch.transpose(outputs.cpu(), 0, 1))]

    # Calcular la precisión en los datos de prueba
    print("Error en el conjunto de test: ", np.array(errors) * target_dev)
    test_precission_mse = mean_squared_error(test_tg.cpu(), outputs.cpu())

    # Imprimir los resultados
    print('Test Loss: {:.4f}'.format(test_loss.item()))
    print('Mean Squared Error: {:.2%}'.format(test_precission_mse))

    print(outputs)
    print(test_tg)
