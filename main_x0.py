import numpy as np
import matplotlib.pyplot as plt

from src.data import Dataset, data_from_file

###########################
#  CONTROL FOR THE FILE  #
###########################
aux_dataset = Dataset.all_fixed
###########################

dataset, _ = data_from_file(aux_dataset, full=True, inverted=True, norm=False)

x_label = [int(i/10) for i in range(0, 201, 10)]

fig, ax = plt.subplots(1, 1, figsize=(12.5, 7))
plt.plot(np.transpose(dataset[0].cpu()))
ax.xaxis.set_ticks(np.arange(len(x_label) * 30, step=30))
ax.xaxis.set_ticklabels(x_label)
plt.xlabel('Tiempo $t$', fontsize=17)
plt.xticks(fontsize=15)
plt.ylabel('Valor de la onda', fontsize=17)
plt.yticks(fontsize=15)
plt.title('Valor de la onda para $l_0 = 0.2$ y $c_0 = 1$', fontsize=17)
plt.show()
