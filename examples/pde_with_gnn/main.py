import numpy as np
import sys
import time
from tqdm import trange

from src.gnn.dataset import get_datasets, get_torch_graph
from src.gnn.buildModel import BuildModel
from src.graph.graph2D import Graph2D
from src.pde.ellipticPdeSolver2D import plot_solution_2d

from examples.pde_with_gnn.pdeGNN import PdeGNN
from examples.pde_with_gnn.problem import Problem, get_node_features, get_edge_features

from examples.discrete_laplacian.main import plot_testing


def create_dataset(n_items):
    """
    Create the dataset to use for training the network.
    :param n_items: number of element in the dataset
    :type n_items: int
    """
    print("Creating dataset...")

    for i in trange(n_items, ncols=100, file=sys.stdout):
        n = np.random.randint(25, 50)
        g = Graph2D.get_default_graph(n_vertexes=n)
        pde = Problem(g)
        pde.solve()
        in_feat = get_node_features(pde.mesh.get_vertexes().shape[0], g.get_boundary_mask())
        edge_feat = get_edge_features(pde.mesh.get_vertexes(), pde.mesh.get_full_edges())
        pde.save_object(f'examples/pde_with_gnn/dataset/pde_{i}', in_feat, edge_feat)


def compute_gnn_solution(vert, edges, bound_mask, model):
    """
    Compute the solution given by the gnn
    :param vert: numpy array of shape (n_vertexes, n_dim)
    :type vert: np.ndarray
    :param edges: numpy array of shape (n_edges, 2). Remember to duplicate edges for undirected graphs
    :type edges: np.ndarray
    :param bound_mask: boolean numpy array, bound_mask[i] is True if i-th vertex is on boundary
    :type bound_mask: np.ndarray
    :param model: GNN model used
    :type model: BuildModel
    :return: numpy array containing the solution of shape (n_vertexes, 1)
    :rtype: np.ndarray
    """
    in_feat = get_node_features(vert.shape[0], bound_mask)
    edge_feat = get_edge_features(vert, edges)
    data = get_torch_graph(vert, edges, in_feat, edge_features=edge_feat)
    out = model(data.x, data.edge_index, data.edge_attr)
    return out.detach().numpy()


def visual_test(graph, model):
    """
    Compare the plot of the solution given by FEM and by GNN of a certain graph
    :param graph: graph on which computes the solution of the pde
    :type graph: Graph2D
    :param model: GNN model used
    :type model: BuildModel
    """
    pde = Problem(graph)
    mesh_vert = pde.mesh.get_vertexes()
    mesh_edges = pde.mesh.get_edges()
    pde.solve()
    gnn_sol = compute_gnn_solution(mesh_vert, pde.mesh.get_full_edges(), graph.get_boundary_mask(), model)
    fem_sol = pde.get_solution()
    rel_errs = abs(gnn_sol - fem_sol) / fem_sol
    print(f"Mean relative error between the 2 solutions is {np.mean(rel_errs):.2e}")
    pde.plot_solution(title="FEM solution", interactive=True)
    print("Close all plots to continue...")
    plot_solution_2d(mesh_vert, mesh_edges, gnn_sol, title="GNN solution")


def comparison(model):
    """
    Compare between computing the solution with FEM and with GNN
    :param model: GNN model used
    :type model: BuildModel
    """
    nodes = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 5000, 10000, 20000]
    fem_times, mesh_times, gnn_times = [], [], []
    gnn_errs = []
    for n_vertexes in nodes:
        print(f"\nGraph with {n_vertexes} physical vertices and {n_vertexes + (n_vertexes-1)*4} mesh nodes")
        # Generate graph and signal
        init_time = time.time()
        g = Graph2D.get_default_graph(n_vertexes)
        print(f"Graph built in {(time.time() - init_time):.2e} seconds")
        # Pde with fem
        init_time = time.time()
        pde = Problem(g)
        mesh_times.append(time.time() - init_time)
        print(f"Mesh built in {mesh_times[-1]:.2e} seconds")
        if n_vertexes <= 5000:
            init_time = time.time()
            pde.solve()
            fem_times.append(time.time() - init_time)
            print(f"Solving PDE with FEM, time elapsed is {fem_times[-1]:.2e} seconds")
        else:
            fem_times.append(np.inf)
        # Pde with gnn
        init_time = time.time()
        gnn_out = compute_gnn_solution(pde.mesh.get_vertexes(), pde.mesh.get_full_edges(), g.get_boundary_mask(), model)
        gnn_times.append(time.time() - init_time)
        print(f"Solving PDE with GNN, time elapsed is {gnn_times[-1]:.2e} seconds")

        # Computing error
        if n_vertexes <= 5000:
            fem_out = pde.get_solution()
            rel_errs = abs(gnn_out - fem_out) / fem_out
            gnn_errs.append(np.mean(rel_errs))
            print(f"Mean relative error is {gnn_errs[-1]:.2e}")

    # Compute mesh nodes
    nodes = [n + (n-1)*4 for n in nodes]

    # Plot computation times
    plot_testing(nodes, [fem_times, mesh_times, gnn_times],
                 colors=["blue", "red", "green"],
                 labels=["FEM time", "Build mesh time", "GNN time"],
                 y_label="Seconds", interactive=True,
                 title="Computation time comparison")

    # Plot errors
    plot_testing(nodes[0:len(gnn_errs)], [gnn_errs],
                 colors=["blue"],
                 labels=[""], legend=False,
                 y_label="Relative errors",
                 title="Mean relative errors")


def run_simulation(elems):
    """
    Run the complete simulation for pde neural network.
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

    # get dataset
    train, val, test = get_datasets('examples/pde_with_gnn/dataset/pde', num_el=n_items, batch_size=batch_size,
                                    division=[70, 20, 10])

    model = BuildModel(PdeGNN(), train, val, "examples/pde_with_gnn/best_model")

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

    # Compare the plot of the solution given by FEM and by GNN of a certain graph
    print("\nWe now compare the plot of the solution given by FEM and by GNN. "
          "\nConsider the graph in Figure 1, the other 2 images contain the plot of the 2 solutions")
    v = [[0.5, -0.5], [0.4, 0.1], [1, 0], [0.5, 0.5], [0, 1], [-0.7, 0.3], [-0.7, -0.2], [-0.5, -0.7], [0.1, -0.8],
         [0, 0]]
    e = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [1, 9], [9, 1], [4, 9],
         [9, 4], [4, 5], [5, 4], [6, 9], [9, 6], [7, 9], [9, 7], [7, 8], [8, 7]]
    g = Graph2D(vertexes=v, edges=e)
    g.plot(plot_boundary=True, title="Figure 1: domain graph (in green the boundary vertices)", interactive=True)
    visual_test(g, model)

    # Comparison with size increasing graphs
    print("\n\nTo conclude, we test the behaviour of the network with size increasing graphs")
    comparison(model)
