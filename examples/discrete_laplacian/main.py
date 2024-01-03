import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import trange

from src.graph.graph2D import Graph2D
from src.graph.graphObject import GraphObject
from src.gnn.dataset import get_datasets, get_torch_graph
from src.gnn.buildModel import BuildModel

from examples.discrete_laplacian.discreteLaplacianGNN import DiscreteLaplacianGNN


def create_dataset(n_items):
    """
    Create the dataset to use for training the network.
    :param n_items: number of element in the dataset
    :type n_items: int
    """
    print("Creating dataset...")
    for i in trange(n_items, ncols=100):
        n = np.random.randint(100, 200)
        g = Graph2D.get_default_graph(n_vertexes=n)
        inp = np.random.uniform(0, 1, size=(n, 1))
        out = g.get_discrete_laplacian(inp)
        graph = GraphObject(g.vertexes, g.get_edges(), in_features=inp, out_features=out)
        graph.save(f"examples/discrete_laplacian/dataset/graph_{i}")


def comparison(model):
    """
    Compare the discrete laplacian computation between definition and gnn.
    :param model: GNN model
    :type model: BuildModel
    """
    nodes = [10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    mul_times = []
    std_times = []
    gnn_times = []
    mul_errs = []
    gnn_errs = []
    for n_vertexes in nodes:
        print(f"\nGraph with {n_vertexes} nodes")
        # Generate graph and signal
        init_time = time.time()
        g = Graph2D.get_default_graph(n_vertexes)
        inp = np.random.uniform(0, 1, size=(n_vertexes, 1))
        data = get_torch_graph(g.vertexes, g.get_edges(), inp)
        print(f"Graph built in {(time.time() - init_time):.2e} seconds")
        if n_vertexes <= 20000:
            # Discrete laplacian multiplying Lx
            L = g.get_graph_laplacian()
            init_time = time.time()
            mul_out = L @ inp
            mul_times.append(time.time() - init_time)
            print(f"Matrix multiplication, time elapsed is {mul_times[-1]:.2e} seconds")
        else:
            # Not enough RAM to allocate L matrix :(
            mul_times.append(np.inf)
            mul_out = 0
        # Discrete laplacian using the definition
        init_time = time.time()
        standard_out = g.get_discrete_laplacian(inp)
        std_times.append(time.time() - init_time)
        print(f"Using definition, time elapsed is {std_times[-1]:.2e} seconds")
        # Discrete laplacian using gnn
        init_time = time.time()
        gnn_out = model(data.x, data.edge_index, data.edge_attr)
        gnn_times.append(time.time() - init_time)
        print(f"Using the gnn, time elapsed is {gnn_times[-1]:.2e} seconds")

        # Computing errors
        gnn_out = gnn_out.detach().numpy()
        rel_errs = abs(standard_out - gnn_out) / abs(standard_out)
        gnn_errs.append(np.mean(rel_errs))
        print(f"Definition vs GNN relative error is {gnn_errs[-1]:.2e}")
        rel_errs = abs(standard_out - mul_out) / abs(standard_out)
        if n_vertexes <= 20000:
            mul_errs.append(np.mean(rel_errs))
            print(f"Definition vs multiplication relative error is {mul_errs[-1]:.2e}")
        else:
            mul_errs.append(np.inf)

    # Plot computation times
    plot_testing(nodes, [mul_times, std_times, gnn_times],
                 colors=["blue", "red", "green"],
                 labels=["Multiplication time", "Definition time", "GNN time"],
                 y_label="Seconds", interactive=True,
                 title="Computation time comparison")

    # Plot errors comparison
    plot_testing(nodes, [gnn_errs, mul_errs],
                 colors=["blue", "red"],
                 labels=["Gnn vs definition", "Multiplication vs definition"],
                 y_label="Relative errors", interactive=False,
                 title="Errors comparison")


def plot_testing(nodes, values, colors, labels, y_label, title="", legend=True, interactive=False):
    """
    Function to plot testing properties of graphs with increasing vertices
    :param nodes: numbers of vertices of the graphs
    :type nodes: list
    :param values: list of values to plot
    :type values: list[list]
    :param colors: list of colors to use
    :type colors: list[str]
    :param labels: list of labels to use
    :type labels: list[str]
    :param y_label: label of the y-axis
    :type y_label: str
    :param title: title of the plot
    :param legend: if True, show legend
    :param interactive: if True, enable interactive mode
    """

    w, h = plt.figaspect(0.5)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(1, 1, 1)

    for i, v in enumerate(values):
        ax.plot(nodes, v, color=colors[i], label=labels[i])

    ax.set_xlabel("Nodes")
    ax.set_ylabel(y_label)
    if legend:
        plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.title(title)
    if interactive:
        plt.ion()
    else:
        plt.ioff()
    plt.show()


def run_simulation(elems):
    """
    Run the complete simulation for graph laplacian neural network.
    :param elems: list of dimension 4 containing the following:
                  elems[0] --> if True train the network
                  elems[1] --> if True generate new data and train the network
                  elems[2] --> number of items in the dataset
                  elems[3] --> dimension of the batches
    :type elems: list
    """

    train_model = elems[0]
    create_data = elems[1]
    n_items = elems[2]
    batch_size = elems[3]

    if create_data:
        train_model = True
        create_dataset(n_items)

    # Get dataset
    train, val, test = get_datasets('examples/discrete_laplacian/dataset/graph', batch_size= batch_size,
                                    num_el=n_items, division=[70, 20, 10])

    # Build model
    model = BuildModel(DiscreteLaplacianGNN(), train, val, "examples/discrete_laplacian/best_model")

    print("\nThe model used is:")
    model.summary()

    if train_model:
        print("\nStart training")
        model.train(1000, patience=5)
        model.plot_training()
    else:
        model.load_best()

    # Test on testing set
    model.test(test)

    # Comparison with size increasing graph
    print("\n\nTo conclude, we test the behaviour of the network with size increasing graphs")
    comparison(model)
