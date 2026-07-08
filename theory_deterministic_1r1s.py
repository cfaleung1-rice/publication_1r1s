import numpy as np
from scipy.integrate import solve_ivp

def regime_A(t, k, g, alpha, beta):
    denom = (alpha / beta) * np.exp((alpha + beta) * k * t) + 1.0
    c_S_0 = alpha + beta - (alpha + beta) / denom
    c_R = (alpha + beta) / denom
    c_S_R = np.zeros_like(t)
    return c_S_0, c_R, c_S_R

def regime_B(t, k, g, alpha, beta):
    c_S_0 = alpha + beta * (1.0 - 2.0 * np.exp(-g * t))
    c_R = np.zeros_like(t)
    c_S_R = beta * np.exp(-g * t)
    return c_S_0, c_R, c_S_R

def regime_C(t, k, g, alpha, beta):
    tau = (1.0 / g) * np.log(0.5 * (1.0 + beta / alpha))
    c_S_0 = np.empty_like(t)
    c_R = np.empty_like(t)
    c_S_R = np.empty_like(t)

    early = t < tau
    late = ~early

    c_S_0[early] = 0.0
    c_R[early] = alpha * (1.0 - 2.0 * np.exp(g * t[early])) + beta
    c_S_R[early] = alpha * np.exp(g * t[early])

    c_S_0[late] = (alpha + beta) * (
        1.0 - (alpha + beta) / (2.0 * alpha) * np.exp(-g * t[late])
    )
    c_R[late] = 0.0
    c_S_R[late] = ((alpha + beta) ** 2 / (4.0 * alpha)) * np.exp(-g * t[late])

    return c_S_0, c_R, c_S_R

def get_deterministic_curves(t, k, g, alpha, beta):
    k0 = k * max(alpha, beta)
    if k0 <= 0:
        return np.full_like(t, alpha), np.full_like(t, beta), np.zeros_like(t)

    if g > k0:
        return regime_A(t, k, g, alpha, beta)
    else: return regime_B(t, k, g, alpha, beta) if alpha >= beta else regime_C(t, k, g, alpha, beta)
    
def odes(t, y_state, k, g):
    S0, R, SR = y_state
    return [-k*S0*R + 2*g*SR, -k*S0*R, k*S0*R - g*SR]

def solve_numerical(t, alpha, beta, k, g):
    y0 = [alpha, beta, 0.0]
    sol = solve_ivp(odes, [t[0], t[-1]], y0, args=(k, g),
                    t_eval=t, method='LSODA', rtol=1e-8, atol=1e-10)
    return sol.y[0], sol.y[1], sol.y[2]
