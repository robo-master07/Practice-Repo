"""
2-D steady conduction in a cylinder (r,z): ∇^2 θ = 0, 0≤r≤a, 0≤z≤b
BCs:
1) ∂θ/∂r = -(h/k) θ     at r=a,   0≤z≤b          (radial convection/Robin)
2) ∂θ/∂z = -(he/k) θ    at z=b,   0≤r≤a          (end convection/Robin)
3) θ(r,0) = θ0          at z=0,   0≤r≤a          (isothermal base)

Separation: θ(r,z) = Σ A_n R_n(r) Z_n(z)
Radial eigenproblem ⇒ R_n(r) = J0(λ_n r), with eigenvalues α_n = λ_n a from:
    f(α) = α J1(α) − Bi_r J0(α) = 0,      Bi_r = h a / k
Axial part satisfying Robin at z=b:
    Z_n(z) = cosh(λ_n (b−z)) + (he/(k λ_n)) sinh(λ_n (b−z))

Coefficients A_n chosen to match θ(r,0)=θ0 in a weighted least-squares sense over r∈[0,a]
with weight r (cylindrical area weight), using collocation points.


"""

import numpy as np
import mpmath as mp

# problem inputs
a   = 0.02        # cylinder radius [m]
b   = 0.04        # height [m]
k   = 200.0       # solid conductivity [W/m·K]
h   = 1000.0      # side convection [W/m^2·K]
he  = 500.0       # top convection [W/m^2·K]
theta0 = 10.0     # base excess temperature θ0 [K]

#  radial eigenmodes 
M = 12

#  helpers 
J0 = lambda x: mp.besselj(0, x)
J1 = lambda x: mp.besselj(1, x)

Bi_r = h*a/k

def f_alpha(alpha):
    # root condition at r=a: α J1(α) − Bi_r J0(α) = 0
    return alpha*J1(alpha) - Bi_r*J0(alpha)

def find_eigenvalues(M, xmax=200.0, samples=20000):
    
    xs = np.linspace(1e-6, xmax, samples)
    vals = np.array([float(f_alpha(x)) for x in xs])
    roots = []
    for i in range(len(xs)-1):
        if vals[i] == 0.0:
            roots.append(xs[i])
        elif vals[i]*vals[i+1] < 0.0:
            xL, xR = xs[i], xs[i+1]
            try:
                root = mp.findroot(lambda x: f_alpha(x), (xL, xR))
                # de-duplicate close roots
                if all(abs(root - r) > 1e-3 for r in roots):
                    roots.append(float(root))
            except:  
                pass
        if len(roots) >= M:
            break
    return np.array(roots[:M])

def lambda_n(α):
    return α / a

def Z_n(z, lam):
    # axial function that satisfies ∂Z/∂z|_{z=b} = −(he/k) Z(b)
    return mp.cosh(lam*(b - z)) + (he/(k*lam))*mp.sinh(lam*(b - z))

def R_n(r, α):
    # radial Bessel mode J0(λ r) with λ = α/a
    return J0((α/a)*r)

#  eigenvalues 
alphas = find_eigenvalues(M)
lams   = alphas / a

#  fit coefficients A_n at z=0 

Nr = 400
r_nodes = np.linspace(0.0, a, Nr)
w = r_nodes.copy()  # weight ~ r

# Basis matrix B_ij = R_j(r_i) * Z_j(0)
B = np.zeros((Nr, M), dtype=float)
for j, α in enumerate(alphas):
    lam = lams[j]
    Z0 = float(Z_n(0.0, lam))
    B[:, j] = [float(R_n(r, α))*Z0 for r in r_nodes]

# Target vector = θ0 (constant) at z=0
y = np.full(Nr, theta0, dtype=float)

# Weighted least squares: minimize ||W^(1/2)(B A − y)||
W = np.diag(w / (w.max() + 1e-12))
BW = W @ B
yW = W @ y
A_ls, *_ = np.linalg.lstsq(BW, yW, rcond=None)

# grid evaluation 
Nr_plot, Nz_plot = 120, 120
r_grid = np.linspace(0.0, a, Nr_plot)
z_grid = np.linspace(0.0, b, Nz_plot)
theta = np.zeros((Nr_plot, Nz_plot), dtype=float)

for ir, r in enumerate(r_grid):
    #  R(r) for speed
    R_vals = np.array([float(R_n(r, α)) for α in alphas])
    for iz, z in enumerate(z_grid):
        Z_vals = np.array([float(Z_n(z, lam)) for lam in lams])
        theta[ir, iz] = np.dot(A_ls, R_vals * Z_vals)

# checks of BCs
# 1) BC at z=0 should approximate θ0
theta_z0 = theta[:, 0]
err_base = np.max(np.abs(theta_z0 - theta0))

# 2) Robin at r=a: ∂θ/∂r + (h/k) θ ≈ 0
def dtheta_dr_at_a(z_idx):
    # one-sided finite difference at last two r points
    dr = r_grid[-1] - r_grid[-2]
    return (theta[-1, z_idx] - theta[-2, z_idx]) / dr

bc_residual = []
for iz in (0, Nz_plot//2, Nz_plot-1):
    res = dtheta_dr_at_a(iz) + (h/k)*theta[-1, iz]
    bc_residual.append(res)
bc_inf_norm = np.max(np.abs(bc_residual))

print(f"Eigenvalues α_n (first {M}):\n{alphas}")
print(f"Max base BC error |θ(r,0)-θ0| ≈ {err_base:.3e} K")
print(f"Radial Robin residual max ≈ {bc_inf_norm:.3e} K/m")
print("A_n coefficients:\n", A_ls)


