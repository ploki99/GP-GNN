# Graphs generation
To randomly generate a graph, the following options can be specified
* Number of vertices of the graph.
* Dimension of the space (for dimensions 2 and 3, the graph can also be plotted).
* Limits of the bounding box containing the vertices coordinates.
* Minimum number of edges of the graph (this number is ignored if incompatible with the other options set).
* Directed/undirected graph.
* Possibility of having self-loops (only for directed graphs).
* Strong connection (only for directed graph).
* Weak connection, in the case of undirected graphs, coincides with connection property.
* Aciclicity, to force the graph to have no cycles.

In this numerical experiment, we test the code you can use to generate a graph. 
Moreover, you can find how to compute properties from graphs and the comparison of the computation time needed to generate different types of graphs.

You can see the simulation outputs in the [Python notebook](graphs_generation.ipynb) file.
