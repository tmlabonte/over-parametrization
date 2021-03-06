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
epochs = [i for i in range(50, 1050, 50)]
bounds = [[] for i in range(0, 7)]

train_dataset = main.load_data('train', 'CIFAR10', '/hdd/datasets')
val_dataset = main.load_data('val', 'CIFAR10', '/hdd/datasets')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, **kwargs)

for n in range(15, 16):
    nunits = pow(2, n)

    for epoch in epochs:
        model = nn.Sequential(nn.Linear(32 * 32 * nchannels, nunits), nn.ReLU(), nn.Linear(nunits, nclasses))
        model = model.to(device)
        init_model = copy.deepcopy(model)
        optimizer = optim.SGD(model.parameters(), 0.001, momentum=0.9)

        checkpoint_path = "saved_models/CIFAR10/WD0/N" + str(n) + "/E" + str(epoch) + "/checkpoint.pth.tar"
    
        print("Loading checkpoint for model: 2^" + str(n) + " at epoch " + str(epoch))

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        tr_err, tr_loss, tr_margin = main.validate(model, init_model, device, train_loader)
        val_err, val_loss, val_margin = main.validate(model, init_model, device, val_loader)
        measure = measures.calculate(model, init_model, device, train_loader, tr_margin)
        bound = list(measure.items())[-1:]
        bound = float(bound[0][1])
        bounds.append(bound)

#        with open("graph_results.txt", "a") as f:
#            f.write("Hidden units: 2^" + str(n) + " epoch: " + str(epoch) + "\n")
#            f.write('%.3g' % bound)
#            f.write("\n")

#with open("graph_results.txt", "r") as f:
#        for num, line in enumerate(f):
#            if num % 2 == 1:
#                bound[int((num - 1) / 40)].append(float(line))

#plt.plot(epochs, np.array(bound[0]), marker="+", label="Hidden: 2^6", color="blue")
#plt.plot(epochs, np.array(bound[1]), marker="+", label="Hidden: 2^7", color="orange")
#plt.plot(epochs, np.array(bound[2]), marker="+", label="Hidden: 2^8", color="green")
#plt.plot(epochs, np.array(bound[3]), marker="+", label="Hidden: 2^9", color="black")
#plt.plot(epochs, np.array(bound[4]), marker="+", label="Hidden: 2^10", color="brown")
#plt.plot(epochs, np.array(bound[5]), marker="+", label="Hidden: 2^11", color="yellow")
#plt.plot(epochs, np.array(bound[6]), marker="+", label="Hidden: 2^12", color="pink")
#plt.plot(epochs, np.array(bound[7]), marker="+", label="Hidden: 2^13", color="cyan")
#plt.plot(epochs, np.array(bound[8]), marker="+", label="Hidden: 2^14", color="magenta")
plt.plot(epochs, np.array(bounds), marker="+", label="Hidden: 2^15", color="red")
plt.xlabel("Epoch #")
plt.ylabel("Neyshabur '18 Bound")
plt.xticks([i for i in range(0, 1100, 100)])
plt.yscale("log")
plt.title("CIFAR10 - No WD")
plt.legend()
plt.savefig("epoch_vs_bound_no_wd.png")
