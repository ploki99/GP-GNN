import numpy as np
import random
import time

from src.graph.graphProperties import GraphProperties
from src.graph.graph2D import Graph2D
from src.graph.graph3D import Graph3D

from examples.discrete_laplacian.main import plot_testing


def get_vertexes(n_vert):
    """
    Compute n_vert vertexes coordinates in 2D space. Each vertex is on unit circle and equi-distant
    :param n_vert: number of vertexes
    :type n_vert: int
    :return: list of vertexes coordinates
    """
    v = []
    for i in range(n_vert):
        x = np.cos(2 * i * np.pi / n_vert)
        y = np.sin(2 * i * np.pi / n_vert)
        v.append([x, y])
    return v


def comparison():
    """
    Compare the generation time needed by different types of graph for graphs with increasing number of nodes
    """
    print("\n\nTo conclude, we test the creation time for different types of graph")
    nodes = [10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    time1, time2, time3 = [], [], []
    for n_vertexes in nodes:
        print(f"\nGraph with {n_vertexes} nodes")
        # Generic graph
        if n_vertexes <= 20000:
            init_time = time.time()
            prop = GraphProperties(n_vertexes=n_vertexes, directed=True, loop=True, weak_connected=False, acyclic=False)
            Graph2D(prop)
            time1.append(time.time() - init_time)
            print(f"Generic graph created in {time1[-1]:.2e} seconds")
        else:
            time1.append(np.inf)
        # DAG graph
        if n_vertexes <= 1000:
            init_time = time.time()
            prop = GraphProperties(n_vertexes=n_vertexes, directed=True, loop=True, weak_connected=False, acyclic=True)
            Graph2D(prop)
            time2.append(time.time() - init_time)
            print(f"DAG created in {time2[-1]:.2e} seconds")
        else:
            time2.append(np.inf)
        # Default graph
        init_time = time.time()
        Graph2D.get_default_graph(n_vertexes=n_vertexes)
        time3.append(time.time() - init_time)
        print(f"Default graph created in {time3[-1]:.2e} seconds")

    plot_testing(nodes, [time1, time2, time3],
                 colors=["blue", "red", "green"],
                 labels=["Generic graph time", "DAG time", "Default graph time"],
                 y_label="Seconds", title="Generation time comparison")


def run_simulation(elems):
    """
    Run the complete simulation for graphs generation.
    :param elems: list of dimension 1 containing the following:
                  elems[0] --> seed to use for generating the graphs
    :type elems: list
    """
    # Set seed
    seed = elems[0]
    random.seed(seed)
    # Get vertexes
    v = get_vertexes(10)
    # Directed graphs
    print("Let's start with directed graph")
    p = GraphProperties(n_vertexes=10, minimum_edges=8, directed=True, loop=True, weak_connected=False, acyclic=False)
    g = Graph2D(p, v)
    g.plot(interactive=True, title="Generic directed graph with no particular properties")
    p = GraphProperties(n_vertexes=10, minimum_edges=8, directed=True, loop=True, weak_connected=False, acyclic=True)
    g = Graph2D(p, v)
    g.plot(interactive=True, title="Acyclic directed graph")
    p = GraphProperties(n_vertexes=10, minimum_edges=8, directed=True, loop=True, weak_connected=True, acyclic=True)
    g = Graph2D(p, v)
    g.plot(interactive=True, title="Weak connected and acyclic directed graph")
    p = GraphProperties(n_vertexes=10, minimum_edges=8, directed=True, loop=True, strong_connected=True, acyclic=True)
    g = Graph2D(p, v)
    print("Close all plots to continue...")
    g.plot(interactive=False, title="Strong connected directed graph (it can't be acyclic)")
    # Undirected graph
    print("\nNow, we focus on undirected graph generation")
    p = GraphProperties(n_vertexes=10, minimum_edges=8, directed=False, weak_connected=False, acyclic=False)
    g = Graph2D(p, v)
    g.plot(interactive=True, title="Generic undirected graph")
    p = GraphProperties(n_vertexes=10, minimum_edges=8, directed=False, weak_connected=False, acyclic=True)
    g = Graph2D(p, v)
    g.plot(interactive=True, title="Acyclic undirected graph")
    p = GraphProperties(n_vertexes=10)
    g = Graph2D(p, v)
    print("Close all plots to continue...")
    g.plot(interactive=False, title="Default graph: undirected, acyclic and connected")
    # 3D space
    print("\nOur code is able also to plot graph in 3D space")
    p = GraphProperties(n_vertexes=10, minimum_edges=8, directed=True, weak_connected=False, acyclic=True)
    g = Graph3D(p)
    g.plot(interactive=True, title="DAG in 3D")
    g = Graph3D.get_default_graph(n_vertexes=10)
    print("Close all plots to continue...")
    g.plot(interactive=False, title="Default options in 3D space")
    # More options
    print("\n\nIt's also possible to let the code understand the type of graph, given the matrix of the vertex "
          "coordinates and the edge matrix")
    vertexes = get_vertexes(5)
    edges = [[3, 3], [0, 2], [1, 0], [1, 2], [2, 2], [2, 0], [3, 0], [2, 1], [4, 0], [4, 4]]
    p = GraphProperties.get_properties_from_graph(vertexes, edges)
    g = Graph2D(vertexes=vertexes, edges=edges)
    g.plot(interactive=True, title="Plot 1")
    print("\nReferring to plot 1, the graph has the following properties")
    print(p)
    edges = [[0, 3], [3, 0], [1, 0], [0, 1], [1, 2], [2, 1], [2, 0], [0, 2], [4, 0], [0, 4]]
    p = GraphProperties.get_properties_from_graph(vertexes, edges)
    g = Graph2D(vertexes=vertexes, edges=edges)
    print("\nReferring to plot 2, the graph has the following properties")
    print(p)
    print("\nClose all plots to continue...")
    g.plot(interactive=False, title="Plot 2")

    comparison()
