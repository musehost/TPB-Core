# Benjamin Langston Cortes - 2026
# TPB t' inercial v1.0
# t' = t * (1 - I·r) | I = lp * grad(theta)
# rho_u < rho_P -> t' -> 0

import numpy as np

def rho_Q(psi_list, lp=1.616e-35, hbar=1.0545718e-34):
    V_cell = lp**3
    return (hbar / V_cell) * np.sum(np.abs(psi_list)**2)

def QMap(rho_Q_grid, A_control, mu0=1.25663706e-6):
    rho_k = np.fft.fftn(rho_Q_grid)
    G_k = 1.0 / (1.0 + np.abs(rho_k) * 1e-96)
    geom = np.fft.ifftn(rho_k * G_k).real
    manip_EM = A_control * (1.0 / mu0) * 1e-6
    return {"geometria": geom, "campo_EM": manip_EM}

def QMap_QCD_step(q, gluon_field, rho_Q_local, gs=1.0, m=1.0, dt=1e-90, c=3e8, hbar=1.0545718e-34):
    D_q = np.gradient(q)[0] + 1j * gs * gluon_field * q
    phi_QMap = gs * rho_Q_local * 1e-98
    H_q = -1j * hbar * c * D_q + m * c**2 * q + phi_QMap * q
    q_new = q - (1j / hbar) * H_q * dt
    return q_new

def calcular_t_prima(theta_grid, lp=1.616e-35):
    gx, gy, gz = np.gradient(theta_grid, lp)
    I_mag = lp * np.sqrt(gx**2 + gy**2 + gz**2)
    t_prime = np.clip(1.0 - I_mag, 0.001, 1.0)
    return t_prime, I_mag

if __name__ == "__main__":
    psi = [0.6, 0.4]
    rho = rho_Q(psi)
    print(f"1. rho_Q = {rho:.3e} J/m^3")

    rho_grid = np.ones((10,10,10)) * rho
    A_control = np.zeros((10,10,10,3))
    A_control[5,5,5] = [0,0,1e-6]
    qmap = QMap(rho_grid, A_control)
    print(f"2. QMap geometria centro: {qmap['geometria'][5,5,5]:.3e}")

    quark = np.array([1.0+0j, 1.0+0j])
    gluon = np.array([0.1, 0.1])
    quark_new = QMap_QCD_step(quark, gluon, rho)
    print(f"3. Quark t+dt: {quark_new[0]}")

    # TPB t' - VERSION ESTABLE
    theta = np.random.uniform(0, 0.1, (10,10,10))
    theta[5,5,5] += 0.2
    tp, I = calcular_t_prima(theta)
    print(f"4. TPB t' centro: {tp[5,5,5]:.4f} * t | I={I[5,5,5]:.4f}")
    print(f" I max en grid: {np.max(I):.4f}")

    if np.max(I) < 1.0:
        print("ESTABLE: rho_u < rho_P -> horizonte OK")
    else:
        print("COLAPSO: I>1 -> t' congelado a 0.001")
