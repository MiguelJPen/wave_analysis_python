import numpy
import numpy as np

import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

import preprocess.divide_data
import preprocess.filter_data
import define_loss
import define_optimizer
import define_model
import IO_model
import test_model


data, target = preprocess.filter_data.load_csv(preprocess.filter_data.Dataset.lhs_100K)

target, target_mean, target_dev = preprocess.filter_data.normalize(target)

train_data, val_data, test_data, train_targets, val_targets, test_targets = preprocess.divide_data.load_data(data,
                                                                                                             target)

net = define_model.create_net()

# Definir la función de pérdida y el optimizador
criterion = define_loss.loss_function()
optimizer = define_optimizer.optimizer(net)

# Entrenar la red
for epoch in range(100):
    running_loss = 0.0

    outputs = net(train_data.unsqueeze(1))

    loss = criterion(outputs, train_targets)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    optimizer.zero_grad()
    print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / 100))

    net.eval()
    with torch.no_grad():
        outputs = net(val_data.unsqueeze(1))
        test_loss = define_loss.loss_function()(outputs, val_targets)
        errors = [mean_absolute_error(x, y) for x, y in
                  zip(torch.transpose(val_targets, 0, 1), torch.transpose(outputs, 0, 1))]
        print(np.array(errors) * target_dev)

    net.train()

print('Finished Training')

# IO_model.save_model(net)

net.eval()
# Evaluar el modelo
with torch.no_grad():
    # Obtener las predicciones para los datos de prueba
    outputs = net(test_data.unsqueeze(1))
    # Calcular la pérdida para los datos de prueba
    test_loss = define_loss.loss_function()(outputs, test_targets)

    errors = [mean_absolute_error(x, y) for x, y in zip(torch.transpose(test_targets, 0, 1), torch.transpose(outputs, 0, 1))]

    # Calcular la precisión en los datos de prueba
    print(errors)
    test_precission_mse = mean_squared_error(test_targets, outputs)

    # Imprimir los resultados
    print('Test Loss: {:.4f}'.format(test_loss.item()))
    print('Mean Squared Error: {:.2%}'.format(test_precission_mse))