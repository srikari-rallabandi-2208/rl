"""
pde_solver.py

Contains the TsiveriotisFernandesSolver class, a Crank–Nicolson-based
Tsiveriotis–Fernandes PDE solver for convertible bond pricing.
"""

import numpy as np
import math


class TsiveriotisFernandesSolver:
    """
    A modular Crank-Nicolson solver for the Tsiveriotis-Fernandes PDE system.

    Attributes:
    -----------
    S_max : float
        The maximum stock price boundary for the PDE grid.
    dx    : float
        Spatial step size in log-space.
    dt    : float
        Time step size in years.

    Methods:
    --------
    solve(Pr, T, sigma, r, d, rc, cv, cp, cfq, tfc):
        Solve the convertible bond price PDE for given parameters,
        returning solution grids U and V plus the number of spatial/time steps (M, N).
    """

    def __init__(self, S_max=450.0, dx=0.005, dt=0.0005):
        """
        Constructor for the TsiveriotisFernandesSolver.

        Parameters:
        -----------
        S_max : float
            Max stock price boundary for the PDE (default=450.0).
        dx : float
            Spatial step size in log-space (default=0.005).
        dt : float
            Time step size in years (default=0.0005).
        """
        self.S_max = S_max
        self.dx = dx
        self.dt = dt

    def solve(self, Pr, T, sigma, r, d, rc, cv, cp, cfq, tfc):
        """
        Solve the Tsiveriotis-Fernandes PDE using Crank-Nicolson.

        Parameters:
        -----------
        Pr    : float - Bond principal/face value
        T     : float - Time to maturity in years
        sigma : float - Volatility in decimal form (e.g., 0.2 for 20%)
        r     : float - Risk-free interest rate
        d     : float - Dividend yield (decimal)
        rc    : float - Credit spread (decimal)
        cv    : float - Conversion price
        cp    : float - Coupon rate (annual)
        cfq   : float - Coupons per year
        tfc   : float - Time fraction until first coupon

        Returns:
        --------
        U_grid : np.ndarray
            The total convertible bond value grid in space/time.
        V_grid : np.ndarray
            The risky bond component grid in space/time.
        M      : int
            Number of spatial steps used.
        N      : int
            Number of time steps used.
        """
        # 1) Determine grid sizes
        M = max(1, int(math.log(self.S_max) / self.dx))
        N = max(1, int(T / self.dt))

        # Actual dx, dt (may differ slightly if M,N are forcibly integers)
        actual_dx = math.log(self.S_max) / M
        actual_dt = T / N

        # 2) Initialize solution grids
        U_grid = np.zeros((M + 1, N + 1))  # total convertible value
        V_grid = np.zeros((M + 1, N + 1))  # risky bond part

        # 3) Determine coupon timing in discrete steps
        #    If cfq*T < 1, there may be no full coupon period
        cp_period = int(N / (cfq * T)) if (cfq * T) > 0 else 0
        cp_offset = 0
        if cp_period > 0:
            cp_offset = int(tfc * cp_period)
        cp_array = np.arange(cp_period + cp_offset, N + 1, cp_period)

        # 4) Terminal conditions (t = T)
        for i in range(1, M):
            S_i = math.exp(i * actual_dx)
            if S_i >= cv:
                # Convert to stock
                U_grid[i, N] = Pr * (S_i / cv)
                V_grid[i, N] = 0
            else:
                # Remain as bond
                U_grid[i, N] = Pr
                V_grid[i, N] = Pr
        U_grid[0, N] = Pr
        V_grid[0, N] = Pr
        U_grid[M, N] = Pr * math.exp(M * actual_dx) / cv
        V_grid[M, N] = 0.0

        # Lower boundary S=0 for earlier times
        for j in range(N):
            discount_factor = (1 + (r + rc) * actual_dt) ** (N - j)
            U_grid[0, j] = Pr / discount_factor
            V_grid[0, j] = Pr / discount_factor

        # 5) PDE coefficients
        D = sigma**2 / 2.0
        drift = r - d - sigma**2 / 2.0
        a_coef = D / (actual_dx**2) - drift / (2 * actual_dx)
        c_coef = D / (actual_dx**2) + drift / (2 * actual_dx)
        b_U = -2 * D / (actual_dx**2) - r
        b_V = -2 * D / (actual_dx**2) - (r + rc)

        # 6) Construct block matrices A (LHS) and B_mat (RHS)
        n_int = M - 1
        A = np.zeros((2 * n_int, 2 * n_int))
        B_mat = np.zeros((2 * n_int, 2 * n_int))

        half_dt = actual_dt / 2.0

        diag_block = np.array([
            [1 - half_dt * b_U,      -half_dt * r],
            [0,                      1 - half_dt * b_V]
        ])
        off_block_lower = np.array([
            [-half_dt * a_coef, 0],
            [0,                 -half_dt * a_coef]
        ])
        off_block_upper = np.array([
            [-half_dt * c_coef, 0],
            [0,                 -half_dt * c_coef]
        ])

        diag_block_rhs = np.array([
            [1 + half_dt * b_U,  half_dt * r],
            [0,                  1 + half_dt * b_V]
        ])
        off_block_lower_rhs = np.array([
            [half_dt * a_coef,  0],
            [0,                 half_dt * a_coef]
        ])
        off_block_upper_rhs = np.array([
            [half_dt * c_coef,  0],
            [0,                 half_dt * c_coef]
        ])

        # Populate A and B_mat
        for i in range(n_int):
            # Main diagonal
            A[2*i:2*i+2, 2*i:2*i+2] = diag_block
            B_mat[2*i:2*i+2, 2*i:2*i+2] = diag_block_rhs

            if i > 0:
                A[2*i:2*i+2, 2*(i-1):2*(i-1)+2] = off_block_lower
                B_mat[2*i:2*i+2, 2*(i-1):2*(i-1)+2] = off_block_lower_rhs

            if i < n_int - 1:
                A[2*i:2*i+2, 2*(i+1):2*(i+1)+2] = off_block_upper
                B_mat[2*i:2*i+2, 2*(i+1):2*(i+1)+2] = off_block_upper_rhs

        # 7) Initialize solution vector X from terminal conditions
        X = np.zeros(2 * n_int)
        for i in range(n_int):
            idx = i + 1
            X[2*i]   = U_grid[idx, N]
            X[2*i+1] = V_grid[idx, N]

        # 8) Backward time-stepping
        for n_ in range(N - 1, -1, -1):
            b_ = B_mat.dot(X)

            # Left boundary
            left_bc = np.array([U_grid[0, n_ + 1],
                                V_grid[0, n_ + 1]])
            b_[0:2] += off_block_lower_rhs.dot(left_bc)

            # Right boundary
            right_bc = np.array([
                U_grid[M, n_ + 1],
                V_grid[M, n_ + 1]
            ])
            b_[-2:] += off_block_upper_rhs.dot(right_bc)

            # Coupon
            ft = 0.0
            for k_ in cp_array:
                if n_ <= k_ < n_ + 1:
                    ft = Pr * cp / cfq
                    break
            for i in range(n_int):
                b_[2*i] += ft

            # Solve A * X_new = b_
            X_new = np.linalg.solve(A, b_)
            X = X_new.copy()

            # Update U_grid, V_grid
            for i in range(n_int):
                idx = i + 1
                U_grid[idx, n_] = X[2*i]
                V_grid[idx, n_] = X[2*i + 1]

            # Early exercise condition
            for i in range(1, M):
                S_i = math.exp(i * actual_dx)
                accrual = 0.0
                if cp_period > 0:
                    # approximate accrued coupon
                    steps_since_coupon = (N - n_) % cp_period
                    accrual = (steps_since_coupon * (Pr * cp / cfq)) / cp_period
                conv_val = (Pr + accrual) * (S_i / cv)
                if U_grid[i, n_] < conv_val:
                    U_grid[i, n_] = conv_val
                    V_grid[i, n_] = 0.0

            # Update boundaries after early exercise
            discount_factor = (1 + (r + rc) * actual_dt) ** (N - n_)
            U_grid[0, n_] = Pr / discount_factor
            V_grid[0, n_] = Pr / discount_factor
            U_grid[M, n_] = Pr * math.exp(M * actual_dx) / cv
            V_grid[M, n_] = 0.0

        return U_grid, V_grid, M, N
