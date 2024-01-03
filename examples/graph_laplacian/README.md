# Graph Laplacian
Consider the problem of solving the graph Laplacian system, that is

$$
  L\mathbf{x} = \mathbf{b}
$$

where $L$ is the Laplacian matrix of an undirected, connected and acyclic graph $G = (V, E)$. The $L$ matrix is not invertible, making the solution of the system hard to  compute. Here, we propose an alternative way to compute this solution, using a Graph Neural Network.

In the file [graphLaplacianGNN.py](graphLaplacianGNN.py), there is the subclassing of our base class of Graph Neural Network, where the parameters of the network are specified.

You can see the simulation outputs in the [Python notebook](graphLaplacian.ipynb) file.
