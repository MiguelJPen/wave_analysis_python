import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import define_loss


def test_model(model, test_data, test_targets):
    # Evaluar el modelo
    with torch.no_grad():
        # Obtener las predicciones para los datos de prueba
        outputs = model(test_data)
        # Calcular la pérdida para los datos de prueba
        test_loss = define_loss.loss_function()(outputs, test_targets)

        test_data_part = []
        test_data_part[0] = [x[0] for x in test_data]
        test_data_part[1] = [x[1] for x in test_data]
        test_data_part[2] = [x[2] for x in test_data]

        print(test_data)
        print(test_data_part)

        outputs_part = []
        outputs_part[0] = [x[0] for x in outputs]
        outputs_part[1] = [x[1] for x in outputs]
        outputs_part[2] = [x[2] for x in outputs]

        # Calcular la precisión en los datos de prueba
        test_precission_r2 = r2_score(test_targets, outputs)
        test_precission_mae = mean_absolute_error(test_targets, outputs)
        test_precission_mse = mean_squared_error(test_targets, outputs)

    # Imprimir los resultados
    print('Test Loss: {:.4f}'.format(test_loss.item()))
    print('R2: {:.2%}'.format(test_precission_r2))
    print('Mean Absolute Error: {:.2}'.format(test_precission_mae))
    print('Mean Squared Error: {:.2%}'.format(test_precission_mse))