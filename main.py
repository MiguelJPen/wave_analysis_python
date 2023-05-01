import numpy as np

import torch
from sklearn.metrics import mean_absolute_error

from preprocess.data import get_data, Dataset
from src.define_loss import loss_function
from src.define_optimizer import optimizer
from src.invertedModels.inCh300 import build_300in_net, build_300in_2out_net
from src.invertedModels.inCh600 import build_600in_net

x0_range = 7.5 - 2.5
l0_range = 5 - 0.2
E0_range = 9 - 1

(train_dt, val_dt, test_dt, train_tg, val_tg, test_tg), target_dev = get_data(Dataset.lhs_50K_t20)

torch.cuda.empty_cache()
net = build_300in_net()

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
    outputs = net(test_dt.unsqueeze(1))
    test_loss = loss_function()(outputs, test_tg)

    errors = [mean_absolute_error(x, y) for x, y in zip(torch.transpose(test_tg.cpu(), 0, 1), torch.transpose(outputs.cpu(), 0, 1))]

    real_error = np.array(errors) * target_dev
    print("Error en el conjunto de test: ", real_error)

    relative_error = real_error / [x0_range, l0_range, E0_range]
    print("Error relativo por parámetro: ", ["{0:.2%}".format(ele) for ele in relative_error])

    # Imprimir los resultados
    print('Test Loss: {:.4f}'.format(test_loss.item()))
