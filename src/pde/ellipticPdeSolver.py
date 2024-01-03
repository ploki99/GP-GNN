from dolfin import *
import logging

from .pdeSolver import PdeSolver


class EllipticPdeSolver(PdeSolver):
    """
    Solve the problem:  -div(grad(u)) + w*u = f
    Default with Neumann condition with null derivative.
    Dirichlet condition can be specified with u0 and u0_boundary:
        - u0 is the boundary function
        - u0_boundary python function that return True if a vertex is in the boundary condition
    """
    def __init__(self, graph, internal_nodes=10):
        super().__init__(graph, internal_nodes)

    def solve(self, w, f, u0=None, u0_boundary=None):
        """
        Solve elliptic problem.
        :param w: dolfin Expression that represents w
        :type w: dolfin.Expression | dolfin.Constant
        :param f: dolfin Expression that represents f
        :type f: dolfin.Expression | dolfin.Constant
        :param u0: Dirichlet boundary function
        :type u0: dolfin.Expression | dolfin.UserExpression | dolfin.Constant
        :param u0_boundary: Python function that return True if a vertex is in the boundary condition
        :type u0_boundary: function
        """
        # Define function space
        V = FunctionSpace(self.mesh.mesh, "CG", 1)
        # Check boundary condition
        bc = None
        if u0 is not None:
            bc = DirichletBC(V, u0, u0_boundary)
        # Define function u and v
        u, v = TrialFunction(V), TestFunction(V)
        # Define weak formulation
        a = inner(w, inner(u, v)) * dx + inner(grad(u), grad(v)) * dx
        L = inner(f, v) * dx
        # Solve the problem
        u = Function(V)
        set_log_level(logging.WARNING)
        solve(a == L, u, bc)
        self.solution = u
