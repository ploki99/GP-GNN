from dolfin import Expression, Constant
import numpy as np

from src.graph.graph2D import Graph2D
from src.graph.graphProperties import GraphProperties
from src.pde.ellipticPdeSolver2D import EllipticPdeSolver2D


def run_simulation():
    p = GraphProperties(n_vertexes=5)
    v = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
    e = [[0, 1], [0, 2], [0, 3], [0, 4]]
    g = Graph2D(p, vertexes=v, edges=e)
    g.plot(interactive=True, title="Domain of the problems")
    problems = [Problem1(g), Problem2(g), Problem3(g)]
    for i, pde in enumerate(problems):
        print(f"Solving problem {i + 1}")
        pde.solve()
        print(f"Infinite norm error for problem {i + 1} is {pde.inf_error(pde.exact_sol):.2e}")
        pde.plot_solution(f"Computed solution of problem {i + 1}", interactive=(i+1 < len(problems)))


class Problem1(EllipticPdeSolver2D):
    def __init__(self, graph):
        super().__init__(graph)

    def solve(self, *args, **kwargs):
        # Define f and w
        f = Expression("5 * ( x[0]*x[0] + x[1]*x[1] + 1 )", degree=2)
        w = Expression("x[0]*x[0] + x[1]*x[1] + 1", degree=2)
        # Solve the problem
        super().solve(w, f)

    @staticmethod
    def exact_sol(x):
        return 5


class Problem2(EllipticPdeSolver2D):
    def __init__(self, graph):
        super().__init__(graph)

    def solve(self, *args, **kwargs):
        # Define Dirichlet boundary conditions
        u0 = Expression("x[0]+x[1]", degree=2)

        def u0_boundary(x):
            tol = 1E-15
            for vertex in self.boundaryVertexes:
                if max(abs(vertex - x)) < tol:
                    return True
            return False

        # Define f and w
        f = Expression("(x[0]+x[1]) * ( x[0]*x[0] + x[1]*x[1] + 1 )", degree=2)
        w = Expression("x[0]*x[0] + x[1]*x[1] + 1", degree=2)
        # Solve the problem
        super().solve(w, f, u0, u0_boundary)

    @staticmethod
    def exact_sol(x):
        return x[0] + x[1]


class Problem3(EllipticPdeSolver2D):
    def __init__(self, graph):
        super().__init__(graph, internal_nodes=50)

    def solve(self, *args, **kwargs):
        # Define Dirichlet boundary conditions
        u0 = Constant(0)

        def u0_boundary(x):
            tol = 1E-15
            for vertex in self.boundaryVertexes:
                if max(abs(vertex - x)) < tol:
                    return True
            return False

        # Define f and w
        f = Expression("(4*pi*pi+1)*sin(2*pi*x[0])", degree=2)
        w = Constant(1)
        # Solve the problem
        super().solve(w, f, u0, u0_boundary)

    @staticmethod
    def exact_sol(x):
        return np.sin(2 * np.pi * x[0])
