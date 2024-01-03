import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .graphND import GraphND

matplotlib.use('TkAgg')


class Graph2D(GraphND):
    """
    Extend GraphND class in the case of 2 dimensional space adding plot method.
    """

    def __init__(self, properties=None, vertexes=None, edges=None):
        if properties is not None:
            properties.n_dim = 2
        if vertexes is not None:
            vert = np.array(vertexes)
            if vert.shape[1] != 2:
                raise Exception("Graph2D vertexes matrix must have shape (n_vertexes, 2)")
        super().__init__(properties, vertexes, edges)

    def plot(self, color='blue', r=0.2, interactive=False, show_axis=True, plot_boundary=False, b_color='green',
             title=''):
        """
        Plot a 2D image with graph representation.
        :param color: color used to plot the graph
        :param r: scaling factor
        :param interactive: if True enable interactive mode
        :param show_axis: if False doesn't show axis
        :param plot_boundary: if True plot the boundary vertexes using b_color
        :param b_color: color used to plot the boundary vertexes
        :param title: optional title to plot on the figure
        """
        fig, ax = plt.subplots()

        # Plot edges
        for e in self.edges:
            # Get coordinates
            x0 = self.vertexes[e[0]][0]
            y0 = self.vertexes[e[0]][1]
            x1 = self.vertexes[e[1]][0]
            y1 = self.vertexes[e[1]][1]

            # Plot circle if we have a loop
            if e[0] == e[1] and self.properties.directed:
                theta = np.linspace(0, 1.75 * np.pi, 200)
                x = r * np.cos(theta) + x0 - r / np.sqrt(2)
                y = r * np.sin(theta) + y0 + r / np.sqrt(2)
                ax.plot(x, y, c=color)
                dx1, dy1 = get_dx_dy(x[0], y[0], x1, y1, r / 2)
                ax.arrow(x[0], y[0], x1 + dx1 - x[0], y1 + dy1 - y[0], length_includes_head=True,
                         head_width=0.05, head_length=0.05, fill=True, color=color)

            # Plot line
            else:
                if self.properties.directed:
                    dx1, dy1 = get_dx_dy(x0, y0, x1, y1, r / 2)
                    dx0, dy0 = 0, 0
                    if self._GraphND__is_edge_present([e[1], e[0]]):
                        dx0, dy0 = dx1, dy1
                    ax.arrow(x0 - dx0, y0 - dy0, x1 + dx1 - x0 + dx0, y1 + dy1 - y0 + dy0, length_includes_head=True,
                             head_width=0.05, head_length=0.05, fill=True, color=color)
                else:
                    ax.plot([x0, x1], [y0, y1], c=color)

        # Plot vertexes
        mask = []
        if plot_boundary:
            mask = self.get_boundary_mask()
        for i, v in enumerate(self.vertexes):
            vert_col = color
            if plot_boundary and mask[i]:
                vert_col = b_color
            ax.scatter(v[0], v[1], s=r * 3000, zorder=2, facecolors='white', edgecolors=vert_col)
            ax.text(v[0], v[1], i, fontsize=12, color=vert_col, horizontalalignment='center', verticalalignment='center')

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


def get_dx_dy(x0, y0, x1, y1, rho):
    """
    Only for visualization purposes of the arrows in case of directed graph
    """
    delta = 0
    if x1 == x0:
        theta = np.pi / 2
        delta = np.sign(y0 - y1)
    else:
        theta = np.arctan((y1 - y0) / (x1 - x0))
    dx1 = rho * np.cos(theta) * np.sign(x0 - x1)
    dy1 = rho * np.sin(theta) * np.sign(x0 - x1 + delta) + delta * rho/2
    return dx1, dy1
