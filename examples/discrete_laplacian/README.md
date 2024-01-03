# Discrete Laplacian
This is the first test of our version of Graph Neural Network. Consider the graph Laplacian system

$$
  L\mathbf{x} = \mathbf{b}
$$

where $L$ is the Laplacian matrix of an undirected, connected and acyclic graph $G = (V, E)$.
We aim to compute the vector **b** given the vector **x** and the graph $G$.

This simulation is called *discrete Laplacian* because the vector **b** can be computed using the definition of discrete Laplacian (avoiding the matrix-vector multiplication), i.e.:

$$
  (L\mathbf{x})^{(v)} = b^{(v)} = \sum_{w \in A_v}{x^{(v)} - x^{(w)}} \qquad \forall v \in V
$$

where $A_v$ denotes the set of vertices adjacent to $v$ (i.e. directly connected to $v$ through an edge).

In the file [discreteLaplacianGNN.py](discreteLaplacianGNN.py), there is the subclassing of our base class of Graph Neural Network, where the parameters of the network are specified.

You can see the simulation outputs in the [Python notebook](discreteLaplacian.ipynb) file.
