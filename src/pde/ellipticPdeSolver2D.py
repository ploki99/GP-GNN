import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .ellipticPdeSolver import EllipticPdeSolver

matplotlib.use('TkAgg')


class EllipticPdeSolver2D(EllipticPdeSolver):
    """
    Child class of EllipticPdeSolver that adds method plot.
    """

    def __init__(self, graph, internal_nodes=10):
        super().__init__(graph, internal_nodes)

    def plot_solution(self, title="", interactive=False):
        """
        Plot computed solution.
        :param title: title to plot on the figure
        :param interactive: if True enable interactive mode
        """
        if self.solution is None:
            raise Exception("You have to solve the problem first")

        # Plot solution
        plot_solution_2d(self.mesh.get_vertexes(),
                         self.mesh.get_edges(),
                         self.get_solution(),
                         title,
                         interactive)


def plot_solution_2d(vertexes, edges, sol, title="", interactive=False):
    """
    Plot the solution of the graph pde.
    :param vertexes: numpy array of shape (num_vertexes, 2)
    :type vertexes: np.ndarray
    :param edges: numpy array of shape (num_edges, 2)
    :type edges: np.ndarray
    :param sol: numpy array of shape (num_vertexes, 1) or (num_vertexes, )
    :type sol: np.ndarray
    :param title: title to plot on the figure
    :param interactive: if True set interactive mode
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Flatten solution vector
    sol = sol.ravel()

    legend = 0

    for e in edges:
        p1 = vertexes[e[0]]
        p2 = vertexes[e[1]]
        # Print domain
        lbl = ""
        if legend == 0:
            legend = 1
            lbl = "Domain"
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0], c="red", label=lbl)
        # Print solution
        if legend == 1:
            legend = 2
            lbl = "Solution"
        z1 = sol[e[0]]
        z2 = sol[e[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [z1, z2], c="blue", label=lbl)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend(loc="upper right")
    plt.title(title)
    # Show plot
    if interactive:
        plt.ion()
    else:
        plt.ioff()
    plt.show()
