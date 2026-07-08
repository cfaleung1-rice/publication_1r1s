from math import comb
import numpy as np
import mpmath as mp
from mpmath import mpf, exp, factorial, binomial

###############################################################################
# Analytical Calculation of Moments in Extreme Values of k and g
###############################################################################

def set_dps(dps=50):
    mp.mp.dps = int(dps)

def to_float(x):
    return float(x.real) if hasattr(x, 'real') else float(x)

def h_t(k1, k2, t):
    k1, k2, t = mpf(k1), mpf(k2), mpf(t)
    if k1 == k2:
        return t * exp(-k1 * t)
    return (exp(-k2 * t) - exp(-k1 * t)) / (k1 - k2)

###############################################################################
# Regime A (S3)

def _phi(s, lam_list, y, beta, exclude):
    num = mpf(1)
    for i in range(y, beta+1):
        num *= lam_list[i]
    den = mpf(1)
    for j in range(y, beta+1):
        if lam_list[j] != exclude:
            den *= (s + lam_list[j])
    return num / den

def compute_P_A(t, y, k, alpha, beta, dps=50):
    set_dps(dps)
    t = mpf(t)
    if t == 0:
        return mpf(1) if y == beta else mpf(0)
    N = alpha + beta
    lam = [mpf(k) * mpf(i) * mpf(N - i) for i in range(beta + 1)]
    if y == beta:
        return exp(-lam[beta] * t)
    if y == 0:
        poles = []
        seen = {}
        for i in range(1, beta + 1):
            z = lam[i]
            if z not in seen: seen[z] = []
            seen[z].append(i)
        for z, idxs in seen.items():
            poles.append((z, len(idxs)))
        s = mpf(0)
        tol = mp.power(10, -mp.mp.dps)
        for zeta, W in poles:
            for m in range(1, W + 1):
                if W == 1:
                    A = _phi(-zeta, lam, 1, beta, zeta)
                else:
                    A = _phi(-zeta, lam, 1, beta, zeta) if m == 2 else mp.diff(lambda ss: _phi(ss, lam, 1, beta, zeta), -zeta)
                zt = zeta * t
                inner = mpf(0)
                ell = 0
                while True:
                    term = (zt ** ell) / factorial(m + ell)
                    inner += term
                    if ell > float(zt) and abs(term) < tol * abs(inner):
                        break
                    ell += 1
                s += A * (t ** m) * exp(-zeta * t) * inner
        return s

    poles = []
    seen = {}
    for i in range(y, beta + 1):
        z = lam[i]
        if z not in seen: seen[z] = []
        seen[z].append(i)
    for z, idxs in seen.items():
        poles.append((z, len(idxs)))
    s = mpf(0)
    lam_y = lam[y]
    for zeta, W in poles:
        for m in range(1, W + 1):
            if W == 1:
                A = _phi(-zeta, lam, y, beta, zeta)
            else:
                A = _phi(-zeta, lam, y, beta, zeta) if m == 2 else mp.diff(lambda ss: _phi(ss, lam, y, beta, zeta), -zeta)
            s += (A / factorial(m-1)) * (t ** (m-1)) * exp(-zeta * t)
    return s / lam_y

def precompute_A(k, alpha, beta, dps=50):
    set_dps(dps)
    N = alpha + beta
    lam = [mpf(k) * mpf(i) * mpf(N - i) for i in range(beta + 1)]

    def build_poles(y0):
        seen = {}
        for i in range(y0, beta + 1):
            z = lam[i]
            if z not in seen: seen[z] = []
            seen[z].append(i)
        return [(z, len(idxs)) for z, idxs in seen.items()]

    terms = {}
    for y in range(beta + 1):
        if y == beta:
            terms[y] = ('beta', None)
            continue
        y0 = 1 if y == 0 else y
        coeffs = []
        for zeta, W in build_poles(y0):
            for m in range(1, W + 1):
                if W == 1:
                    A = _phi(-zeta, lam, y0, beta, zeta)
                else:
                    A = _phi(-zeta, lam, y0, beta, zeta) if m == 2 else mp.diff(lambda ss: _phi(ss, lam, y0, beta, zeta), -zeta)
                if y == 0:
                    coeffs.append((zeta, m, A))
                else:
                    coeffs.append((zeta, m, A / factorial(m - 1)))
        terms[y] = ('zero', coeffs) if y == 0 else ('mid', lam[y], coeffs)
    return lam, terms

def compute_P_A_fast(t, y, lam, terms, beta):
    t = mpf(t)
    if t == 0:
        return mpf(1) if y == beta else mpf(0)
    kind = terms[y][0]
    if kind == 'beta':
        return exp(-lam[beta] * t)
    if kind == 'zero':
        _, coeffs = terms[y]
        tol = mp.power(10, -mp.mp.dps)
        s = mpf(0)
        for zeta, m, A in coeffs:
            zt = zeta * t
            inner = mpf(0)
            ell = 0
            while True:
                term = (zt ** ell) / factorial(m + ell)
                inner += term
                if ell > float(zt) and abs(term) < tol * abs(inner):
                    break
                ell += 1
            s += A * (t ** m) * exp(-zeta * t) * inner
        return s

    _, lam_y, coeffs = terms[y]
    s = mpf(0)
    for zeta, m, coeff in coeffs:
        s += coeff * (t ** (m - 1)) * exp(-zeta * t)
    return s / lam_y

def compute_moments_A(k, alpha, beta, t_points, n, dps=50):
    set_dps(dps)
    N = alpha + beta
    lam, terms = precompute_A(k, alpha, beta, dps)
    EXn, EYn, EZn = [], [], []
    for t in t_points:
        t = float(t)
        if t <= 0:
            EXn.append(float(alpha ** n))
            EYn.append(float(beta ** n))
            EZn.append(0.0)
            continue

        Py_all = [compute_P_A_fast(t, y, lam, terms, beta) for y in range(beta + 1)]

        if n == 1:
            ey = 0.0
            for y in range(beta + 1):
                ey += Py_all[y] * y
            ex = N - ey
            EXn.append(ex)
            EYn.append(ey)
            EZn.append(0.0)
            continue

        EY_powers = []
        for kk in range(n + 1):
            s = mpf(0)
            for y in range(beta + 1):
                s += Py_all[y] * (mpf(y) ** kk)
            EY_powers.append(float(s))

        ex = 0.0
        for kk in range(n + 1):
            ex += comb(n, kk) * (N ** (n - kk)) * ((-1)**kk) * EY_powers[kk]

        ey = EY_powers[n]
        EXn.append(ex)
        EYn.append(ey)
        EZn.append(0.0)
    return np.array(EXn), np.array(EYn), np.array(EZn)

def compute_moments_A_fast(k, alpha, beta, t_points, n, dps=50):
    set_dps(dps)
    N = alpha + beta
    lam, terms = precompute_A(k, alpha, beta, dps)
    EXn, EYn, EZn = [], [], []
    for t in t_points:
        t = float(t)
        if t <= 0:
            EXn.append(float(alpha ** n))
            EYn.append(float(beta ** n))
            EZn.append(0.0)
            continue

        Py_all = [None] + [compute_P_A_fast(t, y, lam, terms, beta) for y in range(1, beta + 1)]

        if n == 1:
            ey = 0.0
            for y in range(1, beta + 1):
                ey += Py_all[y] * y
            EXn.append(N - ey)
            EYn.append(ey)
            EZn.append(0.0)
            continue

        EY_powers = []
        for kk in range(n + 1):
            if kk == 0:
                EY_powers.append(1.0)
                continue
            s = mpf(0)
            for y in range(1, beta + 1):
                s += Py_all[y] * (mpf(y) ** kk)
            EY_powers.append(float(s))

        ex = 0.0
        for kk in range(n + 1):
            ex += comb(n, kk) * (N ** (n - kk)) * ((-1) ** kk) * EY_powers[kk]

        EXn.append(ex)
        EYn.append(EY_powers[n])
        EZn.append(0.0)
    return np.array(EXn), np.array(EYn), np.array(EZn)

###############################################################################
# Regime B (S4 - S5)

def lambda_Bi(i, k, alpha, beta):
    return mpf(k) * mpf(alpha - beta + i) * mpf(i)

def compute_kappas_Bi(y, beta, k, alpha, dps=50):
    set_dps(dps)
    kappas = {}
    for j in range(y, beta+1):
        prod = lambda_Bi(j, k, alpha, beta)
        for m in range(y, beta+1):
            if m != j:
                prod *= lambda_Bi(m, k, alpha, beta) / (lambda_Bi(m, k, alpha, beta) - lambda_Bi(j, k, alpha, beta))
        kappas[j] = prod
    return kappas

def compute_P_Bi(t, y, k, alpha, beta, dps=50, kappas=None):
    set_dps(dps)
    t = mpf(t)
    if t == 0:
        return mpf(1) if y == beta else mpf(0)
    if y == 0:
        if kappas is None:
            kappas = compute_kappas_Bi(1, beta, k, alpha, dps)
        s = mpf(0)
        for m in range(1, beta+1):
            lam = lambda_Bi(m, k, alpha, beta)
            s += (kappas[m]/lam) * (1 - exp(-lam*t))
        return s
    else:
        if kappas is None:
            kappas = compute_kappas_Bi(y, beta, k, alpha, dps)
        s = mpf(0)
        lam_y = lambda_Bi(y, k, alpha, beta)
        for m in range(y, beta+1):
            s += kappas[m] * exp(-lambda_Bi(m, k, alpha, beta)*t)
        return s / lam_y

def compute_moments_B(k, g, alpha, beta, t_points, n, dps=50):
    set_dps(dps)
    N = alpha + beta

    kappa_table = {y: compute_kappas_Bi(y, beta, k, alpha, dps) for y in range(1, beta+1)}
    kappas_r = kappa_table[1]   # f_{tau_R} weights over the full {1,...,beta}

    EXn, EYn, EZn = [], [], []
    for tval in t_points:
        t = float(tval)
        if t <= 0:
            EXn.append(float(alpha**n)); EYn.append(float(beta**n)); EZn.append(0.0)
            continue

        preX = preY = preZ = mpf(0)
        for y in range(1, beta+1):
            Py = compute_P_Bi(t, y, k, alpha, beta, dps, kappas=kappa_table[y])
            preX += Py * (mpf(alpha-beta+y)**n)
            preY += Py * (mpf(y)**n)
            preZ += Py * (mpf(beta-y)**n)

        postX = postZ = mpf(0)
        for z in range(beta+1):
            b1 = binomial(beta, z)
            zn = mpf(z)**n
            Xval = mpf(N-2*z)**n
            for q in range(beta-z+1):
                b2 = binomial(beta-z, q)
                sign = (mpf(-1))**q
                gzq = mpf(g)*(z+q)
                for r in range(1, beta+1):
                    kr = kappas_r[r]
                    lam_r = lambda_Bi(r, k, alpha, beta)
                    ht = h_t(gzq, lam_r, t)
                    c = b1 * b2 * sign * kr * ht
                    postZ += zn * c
                    postX += Xval * c

        EXn.append(to_float(preX + postX))
        EYn.append(to_float(preY))
        EZn.append(to_float(preZ + postZ))
    return np.array(EXn), np.array(EYn), np.array(EZn)

def compute_moments_B_fast(k, g, alpha, beta, t_points, n, dps=50):
    set_dps(dps)
    N = alpha + beta
    kappa_table = {y: compute_kappas_Bi(y, beta, k, alpha, dps) for y in range(1, beta + 1)}
    kappas_r = kappa_table[1]
    lam_r = [None] + [lambda_Bi(r, k, alpha, beta) for r in range(1, beta + 1)]
    kr = [None] + [kappas_r[r] for r in range(1, beta + 1)]

    zn = [mpf(z) ** n for z in range(beta + 1)]
    xn = [mpf(N - 2 * z) ** n for z in range(beta + 1)]
    zq = [(z, q, binomial(beta, z) * binomial(beta - z, q) * ((mpf(-1)) ** q))
          for z in range(beta + 1) for q in range(beta - z + 1)]

    EXn, EYn, EZn = [], [], []
    for tval in t_points:
        t = float(tval)
        if t <= 0:
            EXn.append(float(alpha ** n)); EYn.append(float(beta ** n)); EZn.append(0.0)
            continue

        preX = preY = preZ = mpf(0)
        for y in range(1, beta + 1):
            Py = compute_P_Bi(t, y, k, alpha, beta, dps, kappas=kappa_table[y])
            preX += Py * (mpf(alpha - beta + y) ** n)
            preY += Py * (mpf(y) ** n)
            preZ += Py * (mpf(beta - y) ** n)

        H = []
        for s in range(beta + 1):
            gs = mpf(g) * s
            acc = mpf(0)
            for r in range(1, beta + 1):
                acc += kr[r] * h_t(gs, lam_r[r], t)
            H.append(acc)

        postX = postZ = mpf(0)
        for z, q, coef in zq:
            c = coef * H[z + q]
            postZ += zn[z] * c
            postX += xn[z] * c

        EXn.append(to_float(preX + postX))
        EYn.append(to_float(preY))
        EZn.append(to_float(preZ + postZ))
    return np.array(EXn), np.array(EYn), np.array(EZn)

###############################################################################
# Regime C (S6 - S7)

def lambda_Ci(i, k, alpha, beta):
    return mpf(k) * mpf(alpha - beta + i) * mpf(i)

def compute_kappas_Ci(y, beta, k, alpha, dps=50):
    set_dps(dps)
    y_min = beta - alpha
    kappas = {}
    for j in range(max(y, y_min+1), beta+1):
        prod = lambda_Ci(j, k, alpha, beta)
        for m in range(max(y, y_min+1), beta+1):
            if m != j:
                prod *= lambda_Ci(m, k, alpha, beta) / (lambda_Ci(m, k, alpha, beta) - lambda_Ci(j, k, alpha, beta))
        kappas[j] = prod
    return kappas

def compute_P_Ci(t, y, k, alpha, beta, dps=50, kappas=None):
    set_dps(dps)
    t = mpf(t)
    if t == 0:
        return mpf(1) if y == beta else mpf(0)
    y_min = beta - alpha
    if y < y_min:
        return mpf(0)
    elif y == y_min:
        if kappas is None:
            kappas = compute_kappas_Ci(y_min+1, beta, k, alpha, dps)
        s = mpf(0)
        for m in range(y_min+1, beta+1):
            lam = lambda_Ci(m, k, alpha, beta)
            s += (kappas[m]/lam) * (1 - exp(-lam*t))
        return s
    else:
        if kappas is None:
            kappas = compute_kappas_Ci(y, beta, k, alpha, dps)
        s = mpf(0)
        lam_y = lambda_Ci(y, k, alpha, beta)
        for m in range(y, beta+1):
            s += kappas[m] * exp(-lambda_Ci(m, k, alpha, beta)*t)
        return s / lam_y

def get_Y_states_Cii(delta):
    return list(range(0, delta+1, 2)) if delta % 2 == 0 else [0] + list(range(1, delta+1, 2))

def lambda_Cii(y, g, alpha, beta):
    return mpf(g) * mpf(alpha + beta - y) / 2

def compute_kappas_Cii(y, delta, g, alpha, beta, dps=50):
    set_dps(dps)
    Ys = get_Y_states_Cii(delta)
    kappas = {}
    for j in Ys:
        if j < y: continue
        prod = lambda_Cii(j, g, alpha, beta)
        for m in Ys:
            if m >= y and m != j:
                prod *= lambda_Cii(m, g, alpha, beta) / (lambda_Cii(m, g, alpha, beta) - lambda_Cii(j, g, alpha, beta))
        kappas[j] = prod
    return kappas

def compute_P_Cii(t, y, g, alpha, beta, dps=50, kappas=None):
    set_dps(dps)
    t = mpf(t)
    delta = beta - alpha
    Ys = get_Y_states_Cii(delta)
    if y not in Ys:
        return mpf(0)
    if t == 0:
        return mpf(1) if y == delta else mpf(0)
    if y == 0:
        if kappas is None:
            kappas = compute_kappas_Cii(Ys[1] if len(Ys) > 1 else 0, delta, g, alpha, beta, dps)
        s = mpf(0)
        for m in Ys:
            if m > 0:
                lam = lambda_Cii(m, g, alpha, beta)
                s += (kappas.get(m, 0)/lam) * (1 - exp(-lam*t))
        return s
    else:
        if kappas is None:
            kappas = compute_kappas_Cii(y, delta, g, alpha, beta, dps)
        s = mpf(0)
        lam_y = lambda_Cii(y, g, alpha, beta)
        for m in Ys:
            if m >= y:
                s += kappas.get(m, 0) * exp(-lambda_Cii(m, g, alpha, beta)*t)
        return s / lam_y

def compute_moments_C(k, g, alpha, beta, t_points, n, dps=50):
    set_dps(dps)
    N = alpha + beta
    delta = beta - alpha
    y_min = delta
    z_max = N // 2
    Ys_ii = get_Y_states_Cii(delta)

    kappa_ci = {y: compute_kappas_Ci(y, beta, k, alpha, dps) for y in range(y_min+1, beta+1)}
    kappas_ci_r = compute_kappas_Ci(y_min+1, beta, k, alpha, dps)
    kappa_cii = {y: compute_kappas_Cii(y, delta, g, alpha, beta, dps) for y in Ys_ii if y > 0}
    sigma_start = Ys_ii[1] if len(Ys_ii) > 1 else 0
    kappas_cii_R = compute_kappas_Cii(sigma_start, delta, g, alpha, beta, dps)

    EXn, EYn, EZn = [], [], []
    for tval in t_points:
        t = float(tval)
        if t <= 0:
            EXn.append(float(alpha**n)); EYn.append(float(beta**n)); EZn.append(0.0)
            continue

        preX = preY = preZ = mpf(0)
        for y in range(y_min+1, beta+1):
            Py = compute_P_Ci(t, y, k, alpha, beta, dps, kappas=kappa_ci[y])
            preX += Py * (mpf(alpha-beta + y)**n)
            preY += Py * (mpf(y)**n)
            preZ += Py * (mpf(beta - y)**n)

        post_ii_Y = post_ii_Z = mpf(0)
        for y in Ys_ii:
            if y <= 0: continue
            kappas_q = kappa_cii[y]
            inv_lam_y = mpf(1) / lambda_Cii(y, g, alpha, beta)
            yn = mpf(y)**n
            zval_n = (mpf(N - y) / 2)**n
            for q in kappas_q:
                lam_q = lambda_Cii(q, g, alpha, beta)
                kq = kappas_q[q]
                for r in range(y_min+1, beta+1):
                    kr = kappas_ci_r.get(r, 0)
                    lam_r = lambda_Ci(r, k, alpha, beta)
                    ht = h_t(lam_q, lam_r, t)
                    c = inv_lam_y * kq * kr * ht
                    post_ii_Z += zval_n * c
                    post_ii_Y += yn * c

        post_iii_X = post_iii_Z = mpf(0)
        for z in range(z_max+1):
            b1 = binomial(z_max, z)
            zn = mpf(z)**n
            Xval = mpf(N - 2*z)**n
            for q in range(z_max - z + 1):
                b2 = binomial(z_max - z, q)
                sign = (mpf(-1))**q
                gzq = mpf(g)*(z+q)
                for r in kappas_cii_R:
                    kr = kappas_cii_R[r]
                    lam_r = lambda_Cii(r, g, alpha, beta)
                    ht = h_t(gzq, lam_r, t)
                    c = b1 * b2 * sign * kr * ht
                    post_iii_Z += zn * c
                    post_iii_X += Xval * c

        EXn.append(to_float(preX + post_iii_X))
        EYn.append(to_float(preY + post_ii_Y))
        EZn.append(to_float(preZ + post_ii_Z + post_iii_Z))
    return np.array(EXn), np.array(EYn), np.array(EZn)

def compute_moments_C_fast(k, g, alpha, beta, t_points, n, dps=50):
    set_dps(dps)
    N = alpha + beta
    delta = beta - alpha
    y_min = delta
    z_max = N // 2
    Ys = get_Y_states_Cii(delta)

    kappa_ci = {y: compute_kappas_Ci(y, beta, k, alpha, dps) for y in range(y_min + 1, beta + 1)}
    kappas_ci_r = compute_kappas_Ci(y_min + 1, beta, k, alpha, dps)
    kappa_cii = {y: compute_kappas_Cii(y, delta, g, alpha, beta, dps) for y in Ys if y > 0}
    sigma_start = Ys[1] if len(Ys) > 1 else 0
    kappas_cii_R = compute_kappas_Cii(sigma_start, delta, g, alpha, beta, dps)

    Rci = list(range(y_min + 1, beta + 1))
    lam_ci = {r: lambda_Ci(r, k, alpha, beta) for r in Rci}
    Qii = [q for q in Ys if q > 0]
    lam_cii = {q: lambda_Cii(q, g, alpha, beta) for q in Qii}
    inv_lam = {y: mpf(1) / lambda_Cii(y, g, alpha, beta) for y in Qii}
    Riii = list(kappas_cii_R.keys())
    lam_riii = {r: lambda_Cii(r, g, alpha, beta) for r in Riii}
    zn = [mpf(z) ** n for z in range(z_max + 1)]
    xn = [mpf(N - 2 * z) ** n for z in range(z_max + 1)]
    zq = [(z, q, binomial(z_max, z) * binomial(z_max - z, q) * ((mpf(-1)) ** q))
          for z in range(z_max + 1) for q in range(z_max - z + 1)]

    EXn, EYn, EZn = [], [], []
    for tval in t_points:
        t = float(tval)
        if t <= 0:
            EXn.append(float(alpha ** n)); EYn.append(float(beta ** n)); EZn.append(0.0)
            continue

        preX = preY = preZ = mpf(0)
        for y in range(y_min + 1, beta + 1):
            Py = compute_P_Ci(t, y, k, alpha, beta, dps, kappas=kappa_ci[y])
            preX += Py * (mpf(alpha - beta + y) ** n)
            preY += Py * (mpf(y) ** n)
            preZ += Py * (mpf(beta - y) ** n)

        G = {}
        for q in Qii:
            acc = mpf(0)
            for r in Rci:
                acc += kappas_ci_r.get(r, 0) * h_t(lam_cii[q], lam_ci[r], t)
            G[q] = acc

        post_ii_Y = post_ii_Z = mpf(0)
        for y in Qii:
            kq = kappa_cii[y]
            inner = mpf(0)
            for q in kq:
                inner += kq[q] * G[q]
            post_ii_Z += inv_lam[y] * ((mpf(N - y) / 2) ** n) * inner
            post_ii_Y += inv_lam[y] * (mpf(y) ** n) * inner

        H = []
        for s in range(z_max + 1):
            gs = mpf(g) * s
            acc = mpf(0)
            for r in Riii:
                acc += kappas_cii_R[r] * h_t(gs, lam_riii[r], t)
            H.append(acc)

        post_iii_X = post_iii_Z = mpf(0)
        for z, q, coef in zq:
            c = coef * H[z + q]
            post_iii_Z += zn[z] * c
            post_iii_X += xn[z] * c

        EXn.append(to_float(preX + post_iii_X))
        EYn.append(to_float(preY + post_ii_Y))
        EZn.append(to_float(preZ + post_ii_Z + post_iii_Z))
    return np.array(EXn), np.array(EYn), np.array(EZn)

###############################################################################
# Overall

def compute_moments(k, g, alpha, beta, t_points, n, dps=50, fast=False):
    alpha = int(alpha); beta = int(beta)
    if n < 0: raise ValueError("n >= 0")
    ratio = float(g)/float(k) if k > 0 else float('inf')
    if ratio > 1:
        return compute_moments_A(k, alpha, beta, t_points, n) if not fast else compute_moments_A_fast(k, alpha, beta, t_points, n)
    elif ratio < 1:
        if alpha >= beta:
            return compute_moments_B(k, g, alpha, beta, t_points, n, dps) if not fast else compute_moments_B_fast(k, g, alpha, beta, t_points, n)
        else:
            return compute_moments_C(k, g, alpha, beta, t_points, n, dps) if not fast else compute_moments_C_fast(k, g, alpha, beta, t_points, n)
    else:
        raise NotImplementedError('Projection is invalid when k = g.')

###############################################################################
# SSA (S1)
###############################################################################

def gillespie_step(x, y, z, k, g):
    a1 = k * x * y
    a2 = g * z
    a0 = a1 + a2
    if a0 == 0:
        return float('inf'), x, y, z
    r1 = np.random.exponential(1.0 / a0)
    r2 = np.random.uniform(0, a0)
    if r2 < a1:
        return r1, x-1, y-1, z+1
    else:
        return r1, x+2, y, z-1

def simulate_single_trajectory(x0, y0, z0, k, g, t_max, t_points):
    t = 0.0
    x, y, z = x0, y0, z0
    traj_t = [t]
    traj_x = [x]
    traj_y = [y]
    traj_z = [z]
    while t < t_max:
        dt, x, y, z = gillespie_step(x, y, z, k, g)
        if dt == float('inf'):
            break
        t += dt
        traj_t.append(t)
        traj_x.append(x)
        traj_y.append(y)
        traj_z.append(z)
    traj_t = np.array(traj_t)
    traj_x = np.array(traj_x)
    traj_y = np.array(traj_y)
    traj_z = np.array(traj_z)
    idx = np.searchsorted(traj_t, t_points, side='right') - 1
    x_at = np.where(idx >= 0, traj_x[idx], x0)
    y_at = np.where(idx >= 0, traj_y[idx], y0)
    z_at = np.where(idx >= 0, traj_z[idx], z0)
    return x_at, y_at, z_at

def simulate_trajectories(x0, y0, z0, k, g, t_max, n_traj=1000, t_points=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if t_points is None:
        t_points = np.linspace(0, t_max, 100)
    n_pts = len(t_points)
    x_all = np.empty((n_traj, n_pts))
    y_all = np.empty((n_traj, n_pts))
    z_all = np.empty((n_traj, n_pts))
    for i in range(n_traj):
        x_all[i, :], y_all[i, :], z_all[i, :] = simulate_single_trajectory(x0, y0, z0, k, g, t_max, t_points)
    return np.asarray(t_points), x_all, y_all, z_all

def get_statistics(trajs):
    mean = np.mean(trajs, axis=0)
    std = np.std(trajs, axis=0, ddof=1)
    var = np.var(trajs, axis=0, ddof=1)
    se = std / np.sqrt(trajs.shape[0])
    fano = var / mean
    return {'mean': mean, 'std': std, 'var': var, 'se': se, 'fano': fano}

def simulate_extinction_time(x0, y0, z0, k, g, max_time=1e12):
    t = 0.0
    x, y, z = x0, y0, z0
    while y > 0 and t < max_time:
        a1 = k * x * y
        a2 = g * z
        a0 = a1 + a2
        if a0 == 0:
            return max_time
        dt = np.random.exponential(1.0 / a0)
        if np.random.uniform(0, a0) < a1:
            x -= 1
            y -= 1
            z += 1
        else:
            x += 2
            z -= 1
        t += dt
    return t

def get_met(x0, y0, z0, k, g, n_traj=1000, max_time=1e12, seed=None):
    if seed is not None:
        np.random.seed(seed)
    taus = np.empty(n_traj)
    for i in range(n_traj):
        taus[i] = simulate_extinction_time(x0, y0, z0, k, g, max_time)
    return np.mean(taus), np.std(taus)

def simulate_extinction_njit(alpha, beta, k, g):
    t = 0.0
    x = alpha
    y = beta
    z = 0
    while y > 0:
        rate_feed = k * x * y
        rate_grow = g * z
        total_rate = rate_feed + rate_grow
        if total_rate == 0.0: break
        dt = np.random.exponential(1.0 / total_rate)
        t += dt
        if np.random.rand() * total_rate < rate_feed:
            x -= 1
            y -= 1
            z += 1
        else:
            z -= 1
            x += 2
    return t

try:
    from numba import njit
    gillespie_step = njit(gillespie_step)
    simulate_extinction_time = njit(simulate_extinction_time)
    simulate_extinction_njit = njit(simulate_extinction_njit)
except ImportError:
    pass

def get_mean_extinction(alpha, beta, k, g, n_traj=200):
    total_time = 0.0
    for _ in range(n_traj):
        total_time += simulate_extinction_njit(alpha, beta, k, g)
    return total_time / n_traj

def analytical_met_phase_A(alpha, beta, k):
    gamma = alpha + beta
    return sum(1.0 / (k * y * (gamma - y)) for y in range(1, beta + 1))

def analytical_met_phase_B(alpha, beta, k):
    return sum(1.0 / (k * y * (alpha - beta + y)) for y in range(1, beta + 1))

def analytical_met_phase_C(alpha, beta, k, g_vals):
    tau_x = sum(1.0 / (k * x * (beta - alpha + x)) for x in range(1, alpha + 1))
    tau_y_coeff = sum(2.0 / (alpha + beta - y) for y in get_Y_states_Cii(beta - alpha) if y > 0)
    return tau_x + (tau_y_coeff / g_vals)

###############################################################################
# Examples
###############################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    plt.rcParams['figure.dpi'] = 120
    t = np.linspace(0, 2, 500)
    n = 1
    
    # A (original)
    startA1 = datetime.now()
    kA, gA = 1, 10**3
    alphaA, betaA = 25, 25
    spaceA = round((alphaA + betaA) * 0.025)
    EX, EY, EZ = compute_moments(kA, gA, alphaA, betaA, t, n, fast=False)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(t, EX, 'b'); axs[0].set_title(r'$\rm{\mathbb{E}}[X_t]$'); axs[0].set_xlabel(r'$t$')
    axs[1].plot(t, EY, 'r'); axs[1].set_title(r'$\rm{\mathbb{E}}[Y_t]$'); axs[1].set_xlabel(r'$t$')
    axs[2].plot(t, EZ, 'g'); axs[2].set_title(r'$\rm{\mathbb{E}}[Z_t]$'); axs[2].set_xlabel(r'$t$')
    for i in [0, 1, 2]:
        axs[i].set_ylim(-spaceA, alphaA + betaA + spaceA)
    fig.suptitle(rf'Regime A ($k = {kA}$, $g = {gA}$, $\alpha = {alphaA}$, $\beta = {betaA}$)')
    plt.tight_layout(); plt.savefig('stochastic_exampleA.png'); plt.show(); plt.clf()
    np.savetxt('A_X.txt', EX); np.savetxt('A_Y.txt', EY); np.savetxt('A_Z.txt', EZ)
    print(f'Example A (orig) complete: {datetime.now()-startA1}')
    
    # A (fast)
    startA2 = datetime.now()
    kA, gA = 1, 10**3
    alphaA, betaA = 25, 25
    spaceA = round((alphaA + betaA) * 0.025)
    EX, EY, EZ = compute_moments(kA, gA, alphaA, betaA, t, n, fast=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(t, EX, 'b'); axs[0].set_title(r'$\rm{\mathbb{E}}[X_t]$'); axs[0].set_xlabel(r'$t$')
    axs[1].plot(t, EY, 'r'); axs[1].set_title(r'$\rm{\mathbb{E}}[Y_t]$'); axs[1].set_xlabel(r'$t$')
    axs[2].plot(t, EZ, 'g'); axs[2].set_title(r'$\rm{\mathbb{E}}[Z_t]$'); axs[2].set_xlabel(r'$t$')
    for i in [0, 1, 2]:
        axs[i].set_ylim(-spaceA, alphaA + betaA + spaceA)
    fig.suptitle(rf'Regime A ($k = {kA}$, $g = {gA}$, $\alpha = {alphaA}$, $\beta = {betaA}$)')
    plt.tight_layout(); plt.savefig('stochastic_exampleA.png'); plt.show(); plt.clf()
    np.savetxt('A_X.txt', EX); np.savetxt('A_Y.txt', EY); np.savetxt('A_Z.txt', EZ)
    print(f'Example A (fast) complete: {datetime.now()-startA2}')

    # B (original)
    startB1 = datetime.now()
    kB, gB = 10**3, 1
    alphaB, betaB = 30, 20
    spaceB = round((alphaB + betaB) * 0.025)
    EX, EY, EZ = compute_moments(kB, gB, alphaB, betaB, t, n, fast=False)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(t, EX, 'b'); axs[0].set_title(r'$\rm{\mathbb{E}}[X_t]$'); axs[0].set_xlabel(r'$t$')
    axs[1].plot(t, EY, 'r'); axs[1].set_title(r'$\rm{\mathbb{E}}[Y_t]$'); axs[1].set_xlabel(r'$t$')
    axs[2].plot(t, EZ, 'g'); axs[2].set_title(r'$\rm{\mathbb{E}}[Z_t]$'); axs[2].set_xlabel(r'$t$')
    for i in [0, 1, 2]:
        axs[i].set_ylim(-spaceB, alphaB + betaB + spaceB)
    fig.suptitle(rf'Regime B ($k = {kB}$, $g = {gB}$, $\alpha = {alphaB}$, $\beta = {betaB}$)')
    plt.tight_layout(); plt.savefig('stochastic_exampleB.png'); plt.show(); plt.clf()
    np.savetxt('B_X.txt', EX); np.savetxt('B_Y.txt', EY); np.savetxt('B_Z.txt', EZ)
    print(f'Example B (orig) complete: {datetime.now()-startB1}')
    
    # B (fast)
    startB2 = datetime.now()
    kB, gB = 10**3, 1
    alphaB, betaB = 30, 20
    spaceB = round((alphaB + betaB) * 0.025)
    EX, EY, EZ = compute_moments(kB, gB, alphaB, betaB, t, n, fast=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(t, EX, 'b'); axs[0].set_title(r'$\rm{\mathbb{E}}[X_t]$'); axs[0].set_xlabel(r'$t$')
    axs[1].plot(t, EY, 'r'); axs[1].set_title(r'$\rm{\mathbb{E}}[Y_t]$'); axs[1].set_xlabel(r'$t$')
    axs[2].plot(t, EZ, 'g'); axs[2].set_title(r'$\rm{\mathbb{E}}[Z_t]$'); axs[2].set_xlabel(r'$t$')
    for i in [0, 1, 2]:
        axs[i].set_ylim(-spaceB, alphaB + betaB + spaceB)
    fig.suptitle(rf'Regime B ($k = {kB}$, $g = {gB}$, $\alpha = {alphaB}$, $\beta = {betaB}$)')
    plt.tight_layout(); plt.savefig('stochastic_exampleB.png'); plt.show(); plt.clf()
    np.savetxt('B_X.txt', EX); np.savetxt('B_Y.txt', EY); np.savetxt('B_Z.txt', EZ)
    print(f'Example B (fast) complete: {datetime.now()-startB2}')

    # C
    startC1 = datetime.now()
    kC, gC = 10**3, 1
    alphaC, betaC = 20, 30
    spaceC = round((alphaC + betaC) * 0.025)
    EX, EY, EZ = compute_moments(kC, gC, alphaC, betaC, t, n, fast=False)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(t, EX, 'b'); axs[0].set_title(r'$\rm{\mathbb{E}}[X_t]$'); axs[0].set_xlabel(r'$t$')
    axs[1].plot(t, EY, 'r'); axs[1].set_title(r'$\rm{\mathbb{E}}[Y_t]$'); axs[1].set_xlabel(r'$t$')
    axs[2].plot(t, EZ, 'g'); axs[2].set_title(r'$\rm{\mathbb{E}}[Z_t]$'); axs[2].set_xlabel(r'$t$')
    for i in [0, 1, 2]:
        axs[i].set_ylim(-spaceC, alphaC + betaC + spaceC)
    fig.suptitle(rf'Regime C ($k = {kC}$, $g = {gC}$, $\alpha = {alphaC}$, $\beta = {betaC}$)')
    plt.tight_layout(); plt.savefig('stochastic_exampleC.png'); plt.show(); plt.clf()
    np.savetxt('C_X.txt', EX); np.savetxt('C_Y.txt', EY); np.savetxt('C_Z.txt', EZ)
    print(f'Example C (orig) complete: {datetime.now()-startC1}')
    
    # C
    startC2 = datetime.now()
    kC, gC = 10**3, 1
    alphaC, betaC = 20, 30
    spaceC = round((alphaC + betaC) * 0.025)
    EX, EY, EZ = compute_moments(kC, gC, alphaC, betaC, t, n, fast=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(t, EX, 'b'); axs[0].set_title(r'$\rm{\mathbb{E}}[X_t]$'); axs[0].set_xlabel(r'$t$')
    axs[1].plot(t, EY, 'r'); axs[1].set_title(r'$\rm{\mathbb{E}}[Y_t]$'); axs[1].set_xlabel(r'$t$')
    axs[2].plot(t, EZ, 'g'); axs[2].set_title(r'$\rm{\mathbb{E}}[Z_t]$'); axs[2].set_xlabel(r'$t$')
    for i in [0, 1, 2]:
        axs[i].set_ylim(-spaceC, alphaC + betaC + spaceC)
    fig.suptitle(rf'Regime C ($k = {kC}$, $g = {gC}$, $\alpha = {alphaC}$, $\beta = {betaC}$)')
    plt.tight_layout(); plt.savefig('stochastic_exampleC.png'); plt.show(); plt.clf()
    np.savetxt('C_X.txt', EX); np.savetxt('C_Y.txt', EY); np.savetxt('C_Z.txt', EZ)
    print(f'Example C (fast) complete: {datetime.now()-startC2}')