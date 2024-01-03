import numpy as np
import time
from tqdm import trange

from src.graph.graph2D import Graph2D
from src.graph.graphObject import GraphObject
from src.gnn.dataset import get_datasets, get_torch_graph
from src.gnn.buildModel import BuildModel

from examples.graph_laplacian.graphLaplacianGNN import GraphLaplacianGNN
from examples.discrete_laplacian.main import plot_testing


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
        # Create random f s.t sum(f)=0
        x = np.random.uniform(1, 2, (n, 1))
        b = g.get_discrete_laplacian(x)
        graph = GraphObject(g.vertexes, g.get_edges(), in_features=b, out_features=x)
        graph.save(f"examples/graph_laplacian/dataset/graph_{i}")


def comparison(model):
    """
    Consider the graph laplacian problem Lx=b, compare between finding x with GNN and finding x solving ls system
    :param model: GNN model
    :type model: BuildModel
    """
    nodes = [10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    sys_times = []
    gnn_times = []
    gnn_errs = []
    for n_vertexes in nodes:
        print(f"\nGraph with {n_vertexes} nodes")
        # Generate graph and signal
        init_time = time.time()
        g = Graph2D.get_default_graph(n_vertexes)
        x = np.random.uniform(1, 2, size=(n_vertexes, 1))
        b = g.get_discrete_laplacian(x)
        data = get_torch_graph(g.vertexes, g.get_edges(), b)
        print(f"Graph built in {(time.time() - init_time):.2e} seconds")
        if n_vertexes <= 10000:
            # Graph laplacian solving the system
            init_time = time.time()
            L = g.get_graph_laplacian()
            print(f"Graph laplacian matrix built in {(time.time() - init_time):.2e} seconds")
            init_time = time.time()
            np.linalg.lstsq(L, b, rcond=None)
            sys_times.append(time.time() - init_time)
            print(f"Solving least square system, time elapsed is {sys_times[-1]:.2e} seconds")
        else:
            sys_times.append(np.inf)
        # Graph laplacian using gnn
        init_time = time.time()
        gnn_out = model(data.x, data.edge_index, data.edge_attr)
        gnn_times.append(time.time() - init_time)
        print(f"Using the gnn, time elapsed is {gnn_times[-1]:.2e} seconds")

        # Computing error
        gnn_out = gnn_out.detach().numpy()
        rel_errs = abs(gnn_out - x) / x
        gnn_errs.append(np.mean(rel_errs))
        print(f"Mean relative error is {gnn_errs[-1]:.2e}")

    # Plot computation times
    plot_testing(nodes, [sys_times, gnn_times],
                 colors=["blue", "red"],
                 labels=["Solving system time", "GNN time"],
                 y_label="Seconds", interactive=True,
                 title="Computation time comparison")

    # Plot errors
    plot_testing(nodes, [gnn_errs],
                 colors=["blue"],
                 labels=[""], legend=False,
                 y_label="Relative errors",
                 title="Mean relative errors")


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
    train, val, test = get_datasets('examples/graph_laplacian/dataset/graph', batch_size=batch_size,
                                    num_el=n_items, division=[70, 20, 10])

    # Build model
    model = BuildModel(GraphLaplacianGNN(), train, val, "examples/graph_laplacian/best_model")

    print("\nThe model used is:")
    model.summary()

    if train_model:
        print("\nStart training")
        model.train(1000, patience=50)
        model.plot_training()
    else:
        model.load_best()

    # Test on testing set
    model.test(test)

    # Comparison with size increasing graphs
    print("\n\nTo conclude, we test the behaviour of the network with size increasing graphs")
    comparison(model)
