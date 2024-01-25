# GP-GNN
 
Solving graph-defined problems with GNN.

We use Graph Neural Networks to solve problems defined on graph domains 
(from here, the name GP-GNN: graph problems GNN).
In particular, we analyze the resolution of the graph Laplacian system and elliptic PDEs. 
The code is written in Python, and it's organized in the folder [`src`](/src) as follows:
* In [`src/graph`](/src/graph) there is the code used to random generate a graph given its properties,
* In [`src/pde`](/src/pde) there is the implementation of Finite Element Method for graph-defined PDEs,
* Lastly, in [`src/gnn`](/src/gnn) there is the implementation of GNN used for our problems of interest.

Instead, in the [`examples`](/examples) directory there are the [numerical experiments](#numerical-experiments) 
we have done to test our code.

The python module [`main.py`](/main.py) is the main entry of the project, see [how to run the project](#how-to-run-the-project)
for the complete details.

## Table of contents
<!-- TOC -->
* [How to install the project](#how-to-install-the-project)
  * [Main dependencies](#main-dependencies)
  * [Conda environment](#conda-environment)
  * [Troubleshooting](#troubleshooting)
* [How to run the project](#how-to-run-the-project)
* [Numerical experiments](#numerical-experiments)
* [Authors](#authors)
* [License](#license)
<!-- TOC -->

## How to install the project

### Main dependencies
The main dependencies we used are:
* [FEniCS](https://fenicsproject.org/). It is a popular open-source computing platform for solving partial 
  differential equations (PDEs) with the finite element method (FEM).
* For what concerns the implementation of the Neural Network:
  * [PyTorch](https://pytorch.org/): Python package that provides tensor computation and
    deep neural networks built on a tape-based autograd system.
  * [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). It's a library built upon  PyTorch 
    to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.
* [Matplotlib](https://matplotlib.org/): a comprehensive library for creating static, animated, and interactive 
  visualizations in Python.
  
To easily install these dependencies, follow the instruction in the next section to create a conda environment.

### Conda environment 
We developed our code using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
Please verify that you have it properly installed on your system before proceeding. If it's not installed, follow this [instructions](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).

The first step is to create the conda environment with FEniCS and Python version 3.11 
(Of course, you can choose the name you prefer for the environment instead of `GP-GNN`)
```shell
conda create --name GP-GNN -c conda-forge fenics python=3.11
```

Then, you have to activate the environment just created
```shell
conda activate GP-GNN
```

Now, it's time to install the required packages. Let's start with PyTorch
```shell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
Then, PyTorch Geometric 
```shell
conda install pyg -c pyg
```
Now, install torch-summary package, used to plot the summary of a GNN
```shell
pip install torch-summary
```
Afterwards, you have to install Matplotlib
```shell
conda install -c conda-forge matplotlib
```
And finally, the [tqdm](https://github.com/tqdm/tqdm) package, used to create progress bars
```shell
conda install -c conda-forge tqdm
```
**Optionally**, you can install Jupyter to run the notebooks
```shell
pip install jupyter
```
Finally, you can clone this repository and run the project.

### Troubleshooting
If the `conda` command returns `conda: command not found`, follow these steps in order:
1. Verify that conda is actually installed on your system.
2. If you have just installed it, close and reopen the terminal.
3. Verify that you have the `conda` environment variable. To fix it, run the command
   ```shell
   export PATH="/home/username/miniconda3/bin:$PATH"
   ```
   Where you have to replace `/home/username/miniconda3` with your actual path.
   If you want to have the environmental variable when you launch your terminal automatically, use
   ```shell
   conda init
   ```
   which updates the `.bashrc` file.

## How to run the project
First of all, remember to activate the conda environment:
```shell
conda activate GP-GNN
```
Then, you can run the `main.py` module as follows:
* If you want to see the help, use
  ```shell
  python main.py -h
  ```
* If you want to simulate a numerical experiment, execute
  ```shell
  python main.py -r {1,2,3,4,5}
  ```
  where you have to choose the number of the simulation. 
  You can also specify simulation options using `-o` followed by the option you want.
  For the complete list of the possible options and the explanation of what the simulations do, you can run
  ```shell
  python main.py -l
  ```
* A usage example to run the simulation is 
  ```shell
  python main.py -r 3 -o train=True -o data=True
  ```
  where you run simulation number 3 (discrete Laplacian simulation) and 
  you specify to create a new dataset and train again the neural network.

**Warning**: running some simulations may be very time-consuming, so we suggest running with 
the default options (i.e. do not specify any option). Even better, you can look at the 
Python notebook to see the results without losing computation time.

## Numerical experiments

We did the following experiments (the number is the same as that you should use to run the simulation):
1. **Graphs generation**. Generate random graphs given certain properties. 
   You can find the code, the Python notebook, and more information in the folder 
   [`examples/graphs_generation`](/examples/graphs_generation). 
2. **FEM usage**. Test the FEM implementation to solve three PDE problems defined on a star-shaped graph.
  You can find the code, the Python notebook, and more information in the folder 
   [`examples/fem_usage`](/examples/fem_usage). 
3. **Discrete Laplacian**. Compute the discrete Laplacian given an input vector and a graph, using a GNN.
  You can find the code, the Python notebook, and more information in the folder 
   [`examples/discrete_laplacian`](/examples/discrete_laplacian).
4. **Graph Laplacian**. Solve the graph Laplacian system using a GNN.
  You can find the code, the Python notebook, and more information in the folder 
   [`examples/graph_laplacian`](/examples/graph_laplacian).
5. **PDE with GNN**. Solve an elliptic PDE defined on a certain graph using a GNN.
  You can find the code, the Python notebook, and more information in the folder 
   [`examples/pde_with_gnn`](/examples/pde_with_gnn).

## Authors
- Francesca Behrens ([GitHub](https://github.com/francescabehrens))
- Paolo Botta ([GitHub](https://github.com/ploki99))
  
## License

Our code is available under the [MIT License](/LICENSE).
