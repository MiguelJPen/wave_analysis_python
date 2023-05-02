import numpy as np

import torch
from sklearn.metrics import mean_absolute_error

from src.data import get_data, Dataset
from src.invertedModels.define_loss import loss_function
from src.invertedModels.define_optimizer import optimizer
from src import define_model
from torch.utils.data import DataLoader, TensorDataset

x0_range = 7.5 - 2.5
l0_range = 5 - 0.2
E0_range = 9 - 1
BATCH_SIZE = 5120

(train_dt, val_dt, test_dt, train_tg, val_tg, test_tg), target_dev = get_data(Dataset.lhs_50K_t20)

torch.cuda.empty_cache()
net = define_model.create_net_600()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Dispositivo de procesamiento:", device)

train_dataset = TensorDataset(train_dt, train_tg)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(val_dt, val_tg)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

net = net.to(device)
#train_dt, train_tg = train_dt.cuda(), train_tg.cuda()
#val_dt, val_tg = val_dt.cuda(), val_tg.cuda()
#test_dt, test_tg = test_dt.cuda(), test_tg.cuda()

# Definir la función de pérdida y el optimizador
criterion = loss_function()
optim = optimizer(net)

# Entrenar la red
for epoch in range(10000):
    for id_batch, (x_batch, y_batch) in enumerate(train_dataloader):
        running_loss = 0.0
        y_batch_pred = net(x_batch.unsqueeze(1).cuda())

        loss = criterion(y_batch_pred.cuda(), y_batch.cuda())
        loss.backward()
        optim.step()
        running_loss += loss.item()
        optim.zero_grad()

        if epoch % 50 == 49:
            print('Epoch %d loss: %.4f' % (epoch + 1, running_loss / 100))

    net.eval()
    with torch.no_grad():
        for id_batch, (x_batch, y_batch) in enumerate(val_dataloader):
            y_batch_pred = net(x_batch.unsqueeze(1).cuda())

            test_loss = loss_function()(y_batch_pred.cuda(), y_batch.cuda())
            errors = [mean_absolute_error(x, y) for x, y in
                      zip(torch.transpose(y_batch.cpu(), 0, 1), torch.transpose(y_batch_pred.cpu(), 0, 1))]
            if epoch % 50 == 49:
                print("Error en el conjunto de validación: ", np.array(errors) * target_dev)

    net.train()

print('Finished Training')

net.eval()
# Evaluar el modelo
with torch.no_grad():
    outputs = net(test_dt.unsqueeze(1))
    test_loss = loss_function()(outputs, test_tg)

    errors = [mean_absolute_error(x, y) for x, y in zip(torch.transpose(test_tg.cpu(), 0, 1), torch.transpose(outputs.cpu(), 0, 1))]

    # Calcular la precisión en los datos de prueba

    real_error = np.array(errors) * target_dev
    print("Error en el conjunto de test: ", real_error)

    relative_error = real_error / [x0_range, l0_range, E0_range]
    print("Error relativo por parámetro: ", ["{0:.2%}".format(ele) for ele in relative_error])

    # Imprimir los resultados
    print('Test Loss: {:.4f}'.format(test_loss.item()))
