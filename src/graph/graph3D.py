import matplotlib
import matplotlib.pyplot as plt

from .arrow3d import *
from .graphND import GraphND
from .graph2D import get_dx_dy

matplotlib.use('TkAgg')


class Graph3D(GraphND):
    """
    Extend GraphND class in the case of 3 dimensional space adding plot method
    """

    def __init__(self, properties=None, vertexes=None, edges=None):
        if properties is not None:
            properties.n_dim = 3
        if vertexes is not None:
            vert = np.array(vertexes)
            if vert.shape[1] != 3:
                raise Exception("Graph3D vertexes matrix must have shape (n_vertexes, 3)")
        super().__init__(properties, vertexes, edges)

    def plot(self, color='blue', r=0.2, interactive=False, show_axis=True, plot_boundary=False, b_color='green',
             title=''):
        """
        Plot a 3D image with graph representation
        :param color: color used to plot the graph
        :param r: scaling factor
        :param interactive: if True enable interactive mode
        :param show_axis: if False doesn't show axis
        :param plot_boundary: if True plot the boundary vertexes using b_color
        :param b_color: color used to plot the boundary vertexes
        :param title: optional title to plot on the figure
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Plot edges
        for e in self.edges:
            # Get coordinates
            x0 = self.vertexes[e[0]][0]
            y0 = self.vertexes[e[0]][1]
            z0 = self.vertexes[e[0]][2]
            x1 = self.vertexes[e[1]][0]
            y1 = self.vertexes[e[1]][1]
            z1 = self.vertexes[e[1]][2]

            # Plot circle if we have a loop
            if e[0] == e[1] and self.properties.directed:
                theta = np.linspace(0, 1.7 * np.pi, 200)
                x = r * np.cos(theta) + x0 - r / np.sqrt(2)
                y = r * np.sin(theta) + y0 + r / np.sqrt(2)
                z = np.zeros_like(theta) + z0
                ax.plot(x, y, z, c=color)
                # Plot arrow
                dx, dy = get_dx_dy(x[0], y[0], x1, y1, r / 2)
                ax.arrow3D(x[0], y[0], z0, x1 + dx - x[0], y1 + dy - y[0], 0,
                           mutation_scale=15, arrowstyle="-|>", ec=color, fc=color)
            # Plot line
            else:
                if self.properties.directed:
                    # Plot arrow
                    dx1, dy1, dz1 = get_dx_dy_dz(x0, y0, z0, x1, y1, z1, r / 2)
                    dx0, dy0, dz0 = 0, 0, 0
                    if self._GraphND__is_edge_present([e[1], e[0]]):
                        dx0, dy0, dz0 = dx1, dy1, dz1
                    ax.arrow3D(x0 - dx0, y0 - dy0, z0 - dz0,
                               x1 + dx1 - x0 + dx0, y1 + dy1 - y0 + dy0, z1 + dz1 - z0 + dz0,
                               mutation_scale=15, arrowstyle="-|>", ec=color, fc=color)
                else:
                    ax.plot([x0, x1], [y0, y1], [z0, z1], c=color)

        # Plot vertexes
        mask = []
        if plot_boundary:
            mask = self.get_boundary_mask()
        for i, v in enumerate(self.vertexes):
            vert_col = color
            if plot_boundary and mask[i]:
                vert_col = b_color
            ax.scatter(v[0], v[1], v[2], s=r * 2000, facecolors='white', edgecolors=vert_col)
            ax.text(v[0], v[1], v[2], i, fontsize=12, color=vert_col, zorder=self.properties.n_vertexes + 2,
                    horizontalalignment='center', verticalalignment='center')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # Show plot
        if interactive:
            plt.ion()
        else:
            plt.ioff()
        if show_axis:
            plt.axis('on')
        else:
            plt.axis('off')
        plt.title(title)
        plt.show()


def get_dx_dy_dz(x0, y0, z0, x1, y1, z1, rho):
    """
    Only for visualization purposes of the arrows in case of directed graph
    """
    dx1, dy1 = get_dx_dy(x0, y0, x1, y1, rho)
    dx2, dz1 = get_dx_dy(x0, z0, x1, z1, rho)
    dy2, dz2 = get_dx_dy(y0, z0, y1, z1, rho)
    dx = (dx1 + dx2) / 2
    dy = (dy1 + dy2) / 2
    dz = (dz1 + dz2) / 2
    return dx, dy, dz
