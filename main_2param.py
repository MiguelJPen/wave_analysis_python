import numpy as np

import torch
from sklearn.metrics import mean_absolute_error

from preprocess.data import get_data_2param, Dataset
from src.define_loss import loss_function
from src.define_optimizer import optimizer
from src import define_model

l0_range = 5 - 0.2
E0_range = 9 - 1

(train_dt, val_dt, test_dt, train_tg, val_tg, test_tg), target_dev = get_data_2param(Dataset.lhs_x0_7_4_2param)

net = define_model.create_net_300_2output()

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
for epoch in range(10000):
    running_loss = 0.0

    outputs = net(train_dt.unsqueeze(1).cuda())

    loss = criterion(outputs, train_tg)
    loss.backward()
    optim.step()
    running_loss += loss.item()
    optim.zero_grad()

    if epoch % 50 == 49:
        print('Epoch %d loss: %.4f' % (epoch + 1, running_loss / 100))

    net.eval()
    with torch.no_grad():
        outputs = net(val_dt.unsqueeze(1))
        test_loss = loss_function()(outputs, val_tg)
        errors = [mean_absolute_error(x, y) for x, y in
                  zip(torch.transpose(val_tg.cpu(), 0, 1), torch.transpose(outputs.cpu(), 0, 1))]
        if epoch % 50 == 49:
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

    # CALCULAR EL ERROR RELATIVO, DIVIDIENDO ENTRE LA LONG/2

    real_error = np.array(errors) * target_dev
    print("Error en el conjunto de test: ", real_error)

    relative_error = real_error / [l0_range, E0_range]
    print("Error relativo por parámetro: ", ["{0:.2%}".format(ele) for ele in relative_error])

    # Imprimir los resultados
    print('Test Loss: {:.4f}'.format(test_loss.item()))
