import matplotlib.pyplot as plt
import torch
import main
import math
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import measures

device = torch.device("cuda")
nchannels, nclasses = 3, 10
kwargs = {'num_workers': 1, 'pin_memory': True}
hidden_units = np.array([pow(2, n) for n in range(11, 16)])
err = [[] for i in range(0, 4)]
weight_decay = [0, 0.001, 0.0025, 0.005]

train_dataset = main.load_data('train', 'CIFAR10', '/hdd/datasets')
val_dataset = main.load_data('val', 'CIFAR10', '/hdd/datasets')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, **kwargs)

for i, decay in enumerate(weight_decay):
    for nunits in hidden_units:
        n = int(math.log(nunits, 2))

        model = nn.Sequential(nn.Linear(32 * 32 * nchannels, nunits), nn.ReLU(), nn.Linear(nunits, nclasses))
        model = model.to(device)
        init_model = copy.deepcopy(model)
        optimizer = optim.SGD(model.parameters(), 0.001, momentum=0.9, weight_decay=decay)

        checkpoint_path = "saved_models/CIFAR10/SQUARE/WD" + str(decay) + "/N" + str(n) + "/E500/checkpoint.pth.tar"

        print("Loading checkpoint for model: 2^" + str(n) + " at WD: " + str(decay))

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        init_model.load_state_dict(checkpoint['init'])
        
        tr_err, tr_loss, tr_margin = main.validate(model, init_model, device, train_loader, ir_strength=0, square_loss=True)
        val_err, val_loss, val_margin = main.validate(model, init_model, device, val_loader, ir_strength=0, square_loss=True)
        err[i].append(val_err)

plt.plot(hidden_units, np.array(err[0]), marker="+", label="WD: 0", color="blue")
plt.plot(hidden_units, np.array(err[1]), marker="+", label="WD: 0.001", color="black")
plt.plot(hidden_units, np.array(err[2]), marker="+", label="WD: 0.0025", color="green")
plt.plot(hidden_units, np.array(err[3]), marker="+", label="WD: 0.005", color="red")
plt.xlabel("# Hidden Units")
plt.ylabel("Validation Error")
plt.xticks([pow(2, n) for n in range(11, 16)])
plt.xscale("log", basex=2)
plt.title("CIFAR10 - Epochs: 500 Loss: MSE")
plt.legend()
plt.savefig("HU_vs_val_err_square_loss.png")
