import argparse
import importlib


class Simulation:
    """
    Class to handle simulation execution and parameters.
    """
    def __init__(self, name, idx, description, directory):
        """
        :param name: name of the simulation
        :type name: str
        :param idx: index of the simulation
        :type idx: int
        :param description: description of the simulation
        :type description: str
        :param directory: name of the directory containing the simulation
        :type directory: str
        """
        self.name = name
        self.idx = idx
        self.description = description
        self.options = []
        self.options_call = []
        self.directory = directory

    def add_option(self, name, description, default):
        """
        Add a possible option for the simulation.
        :param name: name of the option
        :type name: str
        :param description: description of the option
        :type description: str
        :param default: default value of the option (used to infer variable type)
        :type default: bool | int | float | string
        """
        self.options.append([name, description, default])
        self.options_call.append(default)

    def change_options_call(self, inp):
        """
        Change options to pass to the simulation.
        :param inp: list of raw options read from command line
        :type inp: str
        """
        if inp is None:
            return
        for o in inp:
            vect = o.split("=")
            if len(vect) != 2:
                continue
            self.__change_option(vect[0], vect[1])

    def __change_option(self, name, value):
        """
        Change one option given its name and new value.
        :param name: name of the option
        :type name: str
        :param value: new value of the option
        :type value: bool | int | float | string
        """
        idx = -1
        for i, opt in enumerate(self.options):
            if opt[0] == name:
                idx = i
        if idx == -1:
            return
        # Boolean type
        if type(self.options[idx][2]) == bool:
            if value == "True":
                self.options_call[idx] = True
            elif value == "False":
                self.options_call[idx] = False
        # Integer type
        elif type(self.options[idx][2]) == int:
            self.options_call[idx] = int(value)
        # Float type
        elif type(self.options[idx][2]) == float:
            self.options_call[idx] = float(value)
        # String type
        else:
            self.options_call[idx] = value

    def __str__(self):
        ret = f"\n{self.idx} - {self.name} - {self.description}"
        if len(self.options) > 0:
            ret += "\n\t the following options can be specified:"
            for o in self.options:
                ret += f"\n\t {o[0]:6} -->  {o[1]}. Default {o[0]}={o[2]}"
        else:
            ret += "\n\t no options available for this simulation."
        return ret

    def __call__(self):
        print(f"Starting {self.name} simulation")
        module = importlib.import_module("examples." + self.directory + ".main")
        if len(self.options_call) > 0:
            module.run_simulation(self.options_call)
        else:
            module.run_simulation()


def get_simulations():
    """
    Get list of all possible simulations.
    :return: tuple of list: first list with the simulations, second list only with the indexes
    """
    # Graphs generation simulation
    s1 = Simulation("Graphs generation", 1, "Generate random graphs given properties", "graphs_generation")
    s1.add_option("seed","seed to use for generating the graphs", 42)

    # Fem usage simulation
    s2 = Simulation("Fem usage", 2, "Test code for fem on graph", "fem_usage")

    # Discrete laplacian simulation
    s3 = Simulation("Discrete laplacian", 3, "Execute discrete laplacian simulation", "discrete_laplacian")
    s3.add_option("train", "if True, train the neural network", False)
    s3.add_option("data", "if True, create dataset", False)
    s3.add_option("n", "number of elements in the dataset", 1000)
    s3.add_option("batch", "batch size", 4)

    # Graph laplacian simulation
    s4 = Simulation("Graph laplacian", 4, "Execute graph laplacian simulation", "graph_laplacian")
    s4.add_option("train", "if True, train the neural network", False)
    s4.add_option("data", "if True, create dataset", False)
    s4.add_option("n", "number of elements in the dataset", 500)
    s4.add_option("batch", "batch size", 4)

    # PDE with GNN simulation
    s5 = Simulation("Pde with GNN", 5, "Solve pde defined on graph with GNN", "pde_with_gnn")
    s5.add_option("train", "if True, train the neural network", False)
    s5.add_option("data", "if True, create dataset", False)
    s5.add_option("n", "number of elements in the dataset", 500)
    s5.add_option("batch", "batch size", 4)

    # List of possible simulations
    sim_vect = [s1, s2, s3, s4, s5]
    indexes = [sim.idx for sim in sim_vect]

    return sim_vect, indexes


if __name__ == '__main__':

    # Get possible simulations
    simulations, sim_indexes = get_simulations()

    # Initialize parser
    msg = "Script used to run the simulations contained in the examples folder"
    epilog = "Usage example: main.py -r 3 -o train=True -o data=True --> run discrete laplacian simulation, " \
             "creating new dataset and retraining the network "
    parser = argparse.ArgumentParser(description=msg, epilog=epilog)

    # Add arguments
    parser.add_argument("-l", "--list", action="store_true", help="list all possible simulations")
    parser.add_argument("-o", "--option", action="append", help="option to add for the simulation")
    parser.add_argument("-r", "--run", choices=sim_indexes, help="index of the simulation to run", type=int)
    args = parser.parse_args()

    # Check argument passed
    if args.list:
        print("Possible simulations are:")
        for s in simulations:
            print(s)
    elif args.run:
        simulations[args.run-1].change_options_call(args.option)
        simulations[args.run-1]()
    else:
        print("No simulation selected, run \"main.py -h\" for help")
