from time import time

import numpy as np
import scipy.sparse as sps
from numpy import linspace, zeros, ones, sign, nonzero, reshape, sum, multiply
from numpy.linalg import norm
from scipy.integrate import ode
from scipy.sparse import eye, diags
from scipy.sparse import kron as skron
from scipy.sparse.linalg import spsolve

from .cy.density import moments

def distribution_par(args):
    """Helper function for QuTips parallelisation tool parfor"""
    steps, x, offset, p, tnl, g_1, g_2, l_1, l_2, r_12, r_21, spont, seed = args
    return density(steps, x, offset, p, tnl, g_1, g_2, l_1, l_2, r_12, r_21, spont, seed)


def moments_par(args):
    """Helper function for QuTips parallelisation tool parfor"""
    steps, xw, offset, p, tnl, g_1, g_2, l_1, l_2, r_12, r_21, spont, seed = args
    return moments(steps, xw, offset, p, tnl, g_1, g_2, l_1, l_2, r_12, r_21, spont, seed)


class MasterS:
    """
    Class for solving the MasterEquation, calculatung and storing the data

    Parameters
    ----------
    shape : list
        Dimensions of the Basis of the Masterequation
    A : array_like
        Full sparse matrix representing the Masterequation
    B : array_like
        Sparse matrix on a triangular basis
    rho : array_like
        Steady state solution of the Masterequation in vectorform (Dimension 1)
    RHO_mat : array_like
        Steady state solution of the Masterequation in Matrixform (Dimension 2 or 3)
    ncpu : integer
        Number of CPUs used in multiprocessing
    xw : array_like
        Array of lists in which the Zustands are stored

    scale : float
        Overall time scale of the system
    p : float
        Pumprate of the system
    l_1 : float
        Lossrate of the first mode
    l_2 : float
        Lossrate of the second mode
    g_1 : float
        Gainrate of the first mode
    g_2 : float
        Gainrate of the second mode
    tnl : float
        Lossrate of the carriers
    r12 : float
        Rate of transition from mode 1 to mode 2
    r21 : float
        Rate of transition from mode 2 to mode 1
    spont : float
        Amount of spontaneous Emissions between the modes
    params : str
        Path to the file in which the parameters are stored
    
    Attributes
    ----------
    time : float
        Time spend for the calculation
    solver : str
        Solver used for the calculation
    

    Methods
    -------
    expose()
        Show the parameters of the Masterequation
    show_results(self):
        Show the results of the Masterequation
    create_matrix()
        Set up the matrix for the Masterequation 
    reduce_indices()
        Remove all indices greater then n1 + n2
    set_shape(self, shape, reduceInd=True):
        Set the dimensions of the Masterequation and create the Matrix
    solve()
        Calculate the steady state of the Masterequation with scipy. It is recommended to install scikit-umfpack, for
        UMFPACK-support.
    solve_multigrid()
        Solve the Masterequation with PyAmgs Multigrid-Methods
    solve_mumps()
        Solve the Masterequation with MUMPS solver
    calc_results()
        Obtain the results based on density matrix of the steady state
    monte_carlo(steps, offset=0)
        Prepare the Monte-Carlo-solver and calculate the Correlation functions
    save(self, filename):
        Save the model in npz-Format
    load(self, filename):
        Load the model
    """

    def __init__(self, p=50, params='default'):
        self.scale = 1. / 0.04325518414048072
        self.p = self.scale * p
        self.l_1 = self.scale * 0.00078454
        self.l_2 = self.scale * 0.00140852
        self.g_1 = self.scale * 1e-04
        self.g_2 = 1.4669 * self.g_1
        self.tnl = self.scale * 1. / 23.2512209503
        self.r_21 = 6 * self.scale * 1e-06
        self.r_12 = self.r_21 + self.scale * 0.36592604e-06
        self.spont = 0.

        # Hardcode the Parameters above, or load them
        if not params is 'default':
            loader = np.load('data/' + params)
            self.p = self.scale * p
            self.l_1 = self.scale * loader['l_1']
            self.l_2 = self.scale * loader['l_2']
            self.g_1 = self.scale * loader['tl_1']
            self.g_2 = self.scale * loader['tl_2']
            self.tnl = self.scale * loader['tnl']
            self.r_21 = self.scale * loader['r_21']
            self.r_12 = self.scale * loader['r_12']
            self.spont = loader['spont']

        self.B = None
        self.A = None
        self.x = None
        self.X_amg = None
        self.rho = None
        self.RHO_mat = None
        self.solver = None
        self.time = None
        self.shape = None

        # Amount of used CPUs
        self.ncpu = 8
        self.xw = [np.array([0, 0, 0]) for _ in range(self.ncpu)]
        self.steps = 0

        # Results
        self.n1 = None
        self.n2 = None
        self.g2_1 = None
        self.g2_2 = None
        self.g12 = None

    def expose(self):
        """Show the parameters of the Masterequation """
        print('shape\t:', self.shape)
        print('pump\t:', self.p)

        print('l_1\t:', self.l_1, 'l_2\t:', self.l_2)
        print('g_1\t:', self.g_1, 'g_2\t:', self.g_2)
        print('tn_l\t:', self.tnl)
        print('r_12\t:', self.r_12, 'r_21\t:', self.r_21)
        print('spont\t:', self.spont)

    def show_results(self):
        """Show the results of the Masterequation"""
        print(self.solver)
        print('Time\t:', self.time)
        if not self.B is None:
            print('dyn\t:', self.dyn)
        print('N\tn1\tn2\tg2_1\tg2_2\tg12')
        print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (self.N, self.n1, self.n2, self.g2_1, self.g2_2, self.g12))

    def create_matrix(self):
        """
        Set up the matrix for the Masterequation

        :return:
        """
        n = self.shape
        I = eye(n[0] * n[1] * n[2], n[0] * n[1] * n[2])
        I_N = eye(n[0], n[0])
        I_n1 = eye(n[1], n[1])
        I_n2 = eye(n[2], n[2])

        # Shift n-1
        M_N = skron(diags([ones(n[0] - 1)], [-1], shape=(n[0], n[0])), skron(I_n1, I_n2))
        M_n1 = skron(I_N, skron(diags([ones(n[1] - 1)], [-1], shape=(n[1], n[1])), I_n2))
        M_n2 = skron(I_N, skron(I_n1, diags([ones(n[2] - 1)], [-1], shape=(n[2], n[2]))))

        # Shift n+1
        P_N = skron(diags([ones(n[0] - 1)], [+1], shape=(n[0], n[0])), skron(I_n1, I_n2))
        P_n1 = skron(I_N, skron(diags([ones(n[1] - 1)], [+1], shape=(n[1], n[1])), I_n2))
        P_n2 = skron(I_N, skron(I_n1, diags([ones(n[2] - 1)], [+1], shape=(n[2], n[2]))))

        # Particle number operators
        N_N = skron(diags([range(n[0])], [0], shape=(n[0], n[0])), skron(I_n1, I_n2))
        N_n1 = skron(I_N, skron(diags([range(n[1])], [0], shape=(n[1], n[1])), I_n2))
        N_n2 = skron(I_N, skron(I_n1, diags([range(n[2])], [0], shape=(n[2], n[2]))))

        # Combine particle number operators and shift operators
        MN_N = M_N * N_N
        MN_n1 = M_n1 * N_n1
        MN_n2 = M_n2 * N_n2

        PN_N = P_N * N_N
        PN_n1 = P_n1 * N_n1
        PN_n2 = P_n2 * N_n2

        # Create the Matrix for the mastersystem
        self.A = self.p * (M_N - I) - self.tnl * (N_N - PN_N)
        self.A -= self.g_1 * (N_N * (N_n1 + I) - PN_N * (MN_n1 + M_n1)) + self.g_2 * (
            N_N * (N_n2 + I) - PN_N * (MN_n2 + M_n2))
        self.A -= self.l_1 * (N_n1 - PN_n1) + self.l_2 * (N_n2 - PN_n2)
        self.A -= self.r_12 * (N_n1 * N_n2 - PN_n1 * MN_n2) + self.r_21 * (N_n2 * N_n1 - PN_n2 * MN_n1)

    def reduce_indices(self):
        """
        Remove all indices greater then n1 + n2

        :return:
        """
        n = self.shape

        rho_ind = zeros((n[0], n[1], n[2]))
        if n[1] > n[2]:
            for At in range(n[0]):
                for n2 in range(n[2]):
                    for n1 in range(n[1] - n2):
                        rho_ind[At, n1, n2] = 1
        else:
            for At in range(n[0]):
                for n1 in range(n[1]):
                    for n2 in range(n[2] - n1):
                        rho_ind[At, n1, n2] = 1

        rho_ind = rho_ind.reshape(n[0] * n[1] * n[2])

        self.indices = nonzero(rho_ind)[0]
        self.B = self.A[self.indices][:, self.indices]
        self.x = self.rho[self.indices]
        self.x = self.x / sum(self.x)

    def set_shape(self, shape, reduceInd=True):
        """
        Set the dimensions of the Masterequation and create the Matrix

        :return:
        """
        self.shape = shape
        self.rho = zeros(self.shape[0] * self.shape[1] * self.shape[2])
        self.rho[0] = 1
        self.create_matrix()
        if reduceInd:
            self.reduce_indices()

    def solve(self):
        """
        Calculate the steady state of the Masterequation with scipy. It is recommended to install scikit-umfpack, for
        UMFPACK-support.

        :return:
        """
        startTime = time()

        rhoS = zeros(len(self.rho))

        # Solve the system, UMFPACK is enabled by default
        x = spsolve(self.B, ones(self.x.shape[0]))

        # Normalize the result
        self.x = sign(x) * x / sum(sign(x) * x)

        # Fill in the relevant data
        rhoS[self.indices] = self.x

        # Save results and configuration
        self.time = time() - startTime
        self.rho = rhoS
        self.solver = 'Direct Solver'
        self.calc_results()

    def solve_multigrid(self):
        """
        Solve the Masterequation with PyAmgs Multigrid-Methods

        :return:
        """

        startTime = time()
        rho0 = zeros(len(self.rho))

        # Create start vectors and set up the Matrix for PyAmg
        if not self.X_amg is None:
            B = np.array([self.X_amg]).T;
            BH = B.copy()
        else:
            B = ones((self.B.shape[0], 1), dtype=self.B.dtype);
            BH = B.copy()
        np.random.seed(0)
        b = np.zeros((self.B.shape[0], 1))
        x0 = np.random.rand(self.B.shape[0], 1)

        import pyamg as mlg

        # The configuration was created with PyAmgs solvertester
        ml = mlg.smoothed_aggregation_solver(self.B, B=B, BH=BH,
                                             strength=('evolution', {'epsilon': 2.0, 'k': 2, 'proj_type': 'l2'}),
                                             smooth=('energy', {'weighting': 'local', 'krylov': 'gmres', 'degree': 1,
                                                                'maxiter': 2}),
                                             improve_candidates=None,
                                             aggregate="standard",
                                             presmoother=('gauss_seidel_nr', {'sweep': 'symmetric', 'iterations': 2}),
                                             postsmoother=('gauss_seidel_nr', {'sweep': 'symmetric', 'iterations': 2}),
                                             max_levels=15,
                                             max_coarse=300,
                                             coarse_solver="pinv")

        # Solve system
        x = ml.solve(b, x0=x0, tol=1e-08, accel="gmres", maxiter=400, cycle="V")
        self.X_amg = x

        # Normalize the result
        self.x = sign(x) * x / sum(sign(x) * x)

        # Fill in the relevant data
        rho0[self.indices] = self.x

        # Save results and configuration
        self.time = time() - startTime
        self.rho = rho0
        self.solver = 'Ruge Stuben'
        self.calc_results()

    def solve_mumps(self):
        """
        Solve the Masterequation with MUMPS solver

        :return:
        """
        import petsc4py, sys
        petsc4py.init(sys.argv)
        from petsc4py import PETSc

        startTime = time()

        # Prepare the Matrix for petsc
        A = self.B
        pA = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))

        # create linear solver
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setType('cg')
        ksp.getPC().setType('lu')
        ksp.getPC().setFactorSolverPackage('mumps')

        opts = PETSc.Options()  # dict-like
        opts["mat_mumps_icntl_13"] = 1
        opts["mat_mumps_icntl_24"] = 1
        opts["mat_mumps_cntl_3"] = 1e-12

        # Obtain sol & rhs vectors
        x, b = pA.getVecs()
        x.set(0)
        b.set(1)

        # Solve the system
        ksp.setOperators(pA)
        ksp.setFromOptions()
        ksp.solve(b, x)

        self.time = time() - startTime

        # Normalize the vector and convert to numpy array
        rho = sign(x[...]) * x[...] / sum(sign(x[...]) * x[...])

        self.x = rho
        self.rho[self.indices] = rho
        self.calc_results()

    def calc_results(self):
        """
        Obtain the results based on density matrix of the steady state

        :return:
        """

        n = self.shape
        self.RHO_mat = reshape(self.rho, [n[0], n[1], n[2]])

        self.rhoN = sum(sum(self.RHO_mat, 2), 1)

        self.rho_n1n2 = sum(self.RHO_mat, 0)
        self.rho_Nn1 = sum(self.RHO_mat, 2)
        self.rho_Nn2 = sum(self.RHO_mat, 1)
        self.rho_n1 = sum(self.rho_n1n2, 1)
        self.rho_n2 = sum(self.rho_n1n2, 0)

        nn1 = range(n[1])
        nn2 = range(n[2])
        NN = range(n[0])

        if not self.B is None:
            self.dyn = norm(self.B * self.x)
        self.N = sum(multiply(self.rhoN, NN))
        self.n1 = sum(multiply(self.rho_n1, nn1))
        self.n2 = sum(multiply(self.rho_n2, nn2))
        self.g2_1 = (sum(multiply(self.rho_n1, multiply(nn1, nn1))) - sum(multiply(self.rho_n1, nn1))) / sum(
            multiply(self.rho_n1, nn1)) ** 2

        self.g2_2 = (sum(multiply(self.rho_n2, multiply(nn2, nn2))) - sum(multiply(self.rho_n2, nn2))) / sum(
            multiply(self.rho_n2, nn2)) ** 2
        self.g12 = self.rho_n1n2.dot(nn2).dot(nn1) / (self.n1 * self.n2)

    def monte_carlo(self, steps, offset=0):
        """
        Prepare the Monte-Carlo-solver and calculate the Correlation functions

        :param steps:
            The amount of simulation steps
        :param offset:
            The amount of simulation steps that aren't included in the calculation
        :return:
        """

        # Measure the time of the calculation, for better comparison to other methods
        startTime = time()
        self.steps = steps
        ncpu = self.ncpu

        # Only use QuTips parfor, if more than one CPU is used
        if ncpu > 1:
            from qutip import parfor
            par_args = [(steps / ncpu, self.xw[i], offset, self.p, self.tnl,
                         self.g_1, self.g_2, self.l_1, self.l_2,
                         self.r_12, self.r_21, self.spont,
                         np.random.randint(10000)) for i in range(ncpu)]

            N, n1, n2, nn1, nn2, n12, xw = parfor(moments_par, par_args, num_cpus=ncpu)

            # Save the current states of the simulation fpr all cpus
            xw = xw.astype('int')
            self.xw[:ncpu] = xw

            # Take the average other the results obtained by the different cpus
            self.N, self.n1, self.n2, nn1, nn2, n12 = sum(N) / ncpu, sum(n1) / ncpu, sum(n2) / ncpu, \
                                                      sum(nn1) / ncpu, sum(nn2) / ncpu, sum(n12) / ncpu

        else:
            self.N, self.n1, self.n2, nn1, nn2, n12, self.xw[0] = moments(steps, self.xw[0], offset, self.p, self.tnl,
                                                                          self.g_1, self.g_2,
                                                                          self.l_1, self.l_2, self.r_12, self.r_21,
                                                                          self.spont, np.random.randint(100000000))

        # Calculate the Correlationfunctions and omit the moments of higher order
        self.g2_1 = (nn1 - self.n1) / self.n1 ** 2
        self.g2_2 = (nn2 - self.n2) / self.n2 ** 2
        self.g12 = n12 / (self.n1 * self.n2)

        self.time = time() - startTime
        self.solver = 'Monte Carlo Simulation'

    def save(self, filename):
        """
        Save the model in npz-Format

        :param filename:
            The file
        :return:
        """
        np.savez_compressed(filename,
                 rho=self.rho, shape=self.shape,
                 solver=self.solver, time=self.time,
                 steps=self.steps,
                 N=self.N, n1=self.n1, n2=self.n2,
                 g2_1=self.g2_1, g2_2=self.g2_2, g12=self.g12)

    def load(self, filename):
        """
        Load the model

        :param filename:
        :return:
        """
        loader = np.load(filename + '.npz')
        self.rho = loader['rho']
        self.shape = loader['shape']
        self.solver = loader['solver']
        self.time = loader['time']
        self.steps = loader['steps']
        self.N = loader['N']
        self.n1 = loader['n1']
        self.n2 = loader['n2']
        self.g2_1 = loader['g2_1']
        self.g2_2 = loader['g2_2']
        self.g12 = loader['g12']
