from main import main
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt

# Learning rate is 0.01 for MNIST and 0.001 for CIFAR10
args = {'stopcond': 0, 'dataset': 'CIFAR10', 'learningrate': 0.001, 'nunits': 1024, 'batchsize': 64, 'momentum': 0.9, 'no_cuda': False, 'weightdecay': 0, 'epochs': 400, 'datadir': '/hdd/datasets'}

# vc, l1max, fro, spec_l1, spec_fro, our 
bounds_vals = [[], [], [], [], [], []]
hidden_units = np.array([pow(2, n) for n in range(6, 16)])

for n in range(6, 16):
    args["nunits"] = pow(2, n)

    measure = main(args)
    bounds = list(measure.items())[-6:]
    
    with open("experiment_results.txt", "a") as f:
        f.write("Hidden units: 2^" + str(n) + "\n")
        for key, value in measure.items():
            val = float(value)
            f.write(key + ':\t %.3g' % val)
            f.write("\n")
        f.write("\n")

    for i, bound in enumerate(bounds):
        bounds_vals[i].append(float(bound[1]))

plt.plot(hidden_units, np.array(bounds_vals[0]), marker="+", label="(1) VC-dim", color="blue")
plt.plot(hidden_units, np.array(bounds_vals[1]), marker="+", label="(2) l1,max", color="orange")
plt.plot(hidden_units, np.array(bounds_vals[2]), marker="+", label="(3) Fro", color="green")
plt.plot(hidden_units, np.array(bounds_vals[3]), marker="+", label="(4) spec-l2,1", color="black")
plt.plot(hidden_units, np.array(bounds_vals[4]), marker="+", label="(5) spec-Fro", color="brown")
plt.plot(hidden_units, np.array(bounds_vals[5]), marker="+", label="(6) ours", color="red")
plt.xlabel("# Hidden Units")
plt.ylabel("Capacity")
plt.xscale("log", basex=2)
plt.yscale("log")
plt.xticks([pow(2,6), pow(2,9), pow(2,12), pow(2,15)])
plt.title("CIFAR-10")
plt.legend()
plt.savefig("capacity.png")

plt.clf()

# Get max bound for normalizing
max_val = max(filter(lambda x: isinstance(x, (int, float)), chain.from_iterable(bounds_vals)))

bounds_vals_normalized = [[i / max_val for i in bound] for bound in bounds_vals]

plt.plot(hidden_units, np.array(bounds_vals_normalized[0]), marker="+", label="(1) VC-dim", color="blue")
plt.plot(hidden_units, np.array(bounds_vals_normalized[1]), marker="+", label="(2) l1,max", color="orange")
plt.plot(hidden_units, np.array(bounds_vals_normalized[2]), marker="+", label="(3) Fro", color="green")
plt.plot(hidden_units, np.array(bounds_vals_normalized[3]), marker="+", label="(4) spec-l2,1", color="black")
plt.plot(hidden_units, np.array(bounds_vals_normalized[4]), marker="+", label="(5) spec-Fro", color="brown")
plt.plot(hidden_units, np.array(bounds_vals_normalized[5]), marker="+", label="(6) ours", color="red")
plt.xlabel("# Hidden Units")
plt.ylabel("Normalized Capacity")
plt.xscale("log", basex=2)
plt.xticks([pow(2,6), pow(2,9), pow(2,12), pow(2,15)])
plt.title("CIFAR-10")
plt.legend()
plt.savefig("normalized_capacity.png")