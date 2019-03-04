import matplotlib.pyplot as plt
import torch
import main
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import measures

device = torch.device("cuda")
nchannels, nclasses = 3, 10
kwargs = {'num_workers': 1, 'pin_memory': True}
epochs = np.array([i for i in range(50, 550, 50)])
bounds = [[] for i in range(0, 6)]

train_dataset = main.load_data('train', 'CIFAR10', '/hdd/datasets')
val_dataset = main.load_data('val', 'CIFAR10', '/hdd/datasets')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, **kwargs)

nunits = pow(2, 14)

for epoch in epochs:
    model = nn.Sequential(nn.Linear(32 * 32 * nchannels, nunits), nn.ReLU(), nn.Linear(nunits, nclasses))
    model = model.to(device)
    init_model = copy.deepcopy(model)
    optimizer = optim.SGD(model.parameters(), 0.001, momentum=0.9, weight_decay=0.001)

    checkpoint_path = "saved_models/CIFAR10/WD0.0025/N14/E" + str(epoch) + "/checkpoint.pth.tar"

    print("Loading checkpoint for model: 2^14 at epoch " + str(epoch))

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    init_model.load_state_dict(checkpoint['init'])
    
    tr_err, tr_loss, tr_margin = main.validate(model, init_model, device, train_loader)
    val_err, val_loss, val_margin = main.validate(model, init_model, device, val_loader)
    measure = measures.calculate(model, init_model, device, train_loader, tr_margin)
    bound = list(measure.items())[-6:]
    bound = [float(bound[i][1]) for i in range(0, 6)]
    for i in range(0, 6):
        bounds[i].append(bound[i])

plt.plot(epochs, np.array(bounds[0]), marker="+", label="(1) VC-dim", color="blue")
plt.plot(epochs, np.array(bounds[1]), marker="+", label="(2) l1,max", color="orange")
plt.plot(epochs, np.array(bounds[2]), marker="+", label="(3) Fro", color="green")
plt.plot(epochs, np.array(bounds[3]), marker="+", label="(4) spec-l2,1", color="black")
plt.plot(epochs, np.array(bounds[4]), marker="+", label="(5) spec-Fro", color="brown")
plt.plot(epochs, np.array(bounds[5]), marker="+", label="(6) ours", color="red")
plt.xlabel("Epoch #")
plt.ylabel("Capacity")
plt.xticks([i for i in range(0, 600, 100)])
plt.yscale("log")
plt.title("CIFAR10 - HU: 2^14 WD: 0.0025")
plt.legend()
plt.savefig("epoch_vs_bound_HU214_WD00025.png")
