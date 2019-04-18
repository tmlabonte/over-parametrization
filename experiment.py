from main import train_model
import argparse
import numpy as np
import statistics as s
import matplotlib.pyplot as plt

def main(args):
    # Learning rate is 0.01 for MNIST and 0.001 for CIFAR10
    train_args = {'stopcond': 0, 'dataset': 'CIFAR10', 'learningrate': 0.001, 'nunits': 1024, 'batchsize': 64, 'momentum': 0.9, 'no_cuda': False, 'weightdecay': args["weightdecay"], 'init_reg_strength': args["init_reg_strength"], 'square_loss': args['square_loss'], 'epochs': args["epochs"], 'datadir': '/hdd0/datasets'}

    # vc, l1max, fro, spec_l1, spec_fro, our 
    bounds_vals = [[], [], [], [], [], []]
    total_activations = []
    hidden_units = np.array([pow(2, n) for n in range(6, 16)])

    for n in range(6, 16):
        train_args["nunits"] = pow(2, n)

        measure, activations = train_model(train_args)
        bounds = list(measure.items())[-6:]
        
        for i, a in enumerate(activations):
            activations[i] = a.cpu().numpy()
    
        total_activations.append([a for a in activations])

        with open(args["name"] + ".txt", "a+") as f:
            f.write("Hidden units: 2^" + str(n) + "\n")
            for key, value in measure.items():
                val = float(value)
                f.write(key + ':\t %.3g' % val)
                f.write("\n")
            f.write("\n")

        for i, bound in enumerate(bounds):
            bounds_vals[i].append(float(bound[1]))

    print(avg_zero_activations)
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
    plt.xticks([pow(2, n) for n in range(6, 16, 3)])
    plt.title("CIFAR10 - Epochs: " + str(args["epochs"]) + " WD: " + str(args["weightdecay"]) + " Loss: MSE")
    plt.legend()
    #plt.savefig(args["name"] + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weightdecay", default=0, type=float,
                        help="weight decay (default: 0)")
    parser.add_argument('--init_reg_strength', default=0, type=float,
                        help='initialization regularization strength (default: 0)')
    parser.add_argument("--square_loss", default=False, type=bool,
                        help="activates square loss instead of CE (default: False)")
    parser.add_argument("--epochs", default=500, type=int,
                        help="number of epochs to train (default: 500)")
    parser.add_argument("--name", default="capacity", type=str,
                        help="name of saved .png file (default: capacity)")
    args, unparsed = parser.parse_known_args()
    main(vars(args))
