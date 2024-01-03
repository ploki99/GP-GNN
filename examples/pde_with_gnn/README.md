# Elliptic PDE with GNN
This is the key experiment of our work: we implement a PDE solver with Graph Neural Network. Consider the following problem, defined on an undirected, connected and acyclic graph $G = (V, E)$

$$
  \begin{cases}
    -\Delta u + u = 0 & \quad \text{in } G \\
    u(v) = g & \quad \forall v \in V_D \\
    \sum_{e \in E_v} \frac{\mathrm{d} u}{\mathrm{d} x^e}(v)=0
        & \quad \forall v \in V_N
  \end{cases}
$$
 
where

$$
    g(x,y) = \begin{cases}
        2 & \quad \text{if } x \geq 0 \\
        1 & \quad \text{otherwise}
    \end{cases}
$$

and where we impose Dirichlet conditions on all the boundary vertices.

For this simulation, we use all the previous results: the domain graphs are randomly generated and the "true" solution considered is computed using our implementation of FEM.

In the file [pdeGNN.py](pdeGNN.py), there is the subclassing of our base class of Graph Neural Network, where the parameters of the network are specified. Instead, in the file [problem.py](problem.py), there is the implementation of the FEM problem to be solved. 

You can see the simulation outputs in the [Python notebook](pdeGNN.ipynb) file.
