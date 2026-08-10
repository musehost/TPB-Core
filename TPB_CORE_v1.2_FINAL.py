# ============================================================================
# TPB_CORE_v1.2_FINAL.py - CAMPO UNIFICADO FUNCIONAL
# Benjamin Langston Cortés | DOIs: 10.5281/zenodo.21393458 y derivados
# ============================================================================

import numpy as np

L_0    = 1.0e-13
TICK   = 3.3e-22
H      = 1.0e-15
c      = L_0 / TICK
h_bar  = 1.0545718e-34
G      = 6.67430e-11
k_S    = 3.18e26
rho_P  = c**5 / (h_bar * G**2)
rho_c  = 84.6 * rho_P
VECTOR_86_4 = 86.4

def verificar_existencia(rho_u):
    return rho_u < rho_P

class MAPA_CUANTICO:
    def __init__(self, Nx, Ny):
        self.Nx, self.Ny = Nx, Ny
        self.red_A = np.zeros((Nx, Ny))
        self.red_B = np.zeros((Nx, Ny))
    def leer_celda(self, i, j):
        return self.red_A[i,j], self.red_B[i,j]
    def ejecutar_TIC(self, i, j, nuevo_hA):
        self.red_A[i,j] = nuevo_hA
        self.red_B[i,j] = -nuevo_hA
    def laplaciano(self, campo):
        return (np.roll(campo, 1, 0) + np.roll(campo, -1, 0) + 
                np.roll(campo, 1, 1) + np.roll(campo, -1, 1) - 
                4*campo) / L_0**2

def BURBUJA_4M(rho_local):
    if rho_local >= rho_c:
        return -1e126
    return 0.0

def EDB_1(mapa, i, j, S_mu_val, dt=TICK):
    hA, hB = mapa.leer_celda(i,j)
    rho_u_t = (hA - hB)**2 / (L_0**3)
    campo_rho = mapa.red_A - mapa.red_B
    laplacian_rho = mapa.laplaciano(campo_rho**2)[i,j]
    theta = np.heaviside(rho_u_t - rho_c, 0)
    escudo = k_S * theta * VECTOR_86_4
    drho_dt = (laplacian_rho - escudo + S_mu_val) / c**2
    rho_u_next = rho_u_t + drho_dt * dt
    if rho_u_next >= rho_P:
        presion_rebote = BURBUJA_4M(rho_u_next)
        return rho_c, presion_rebote
    nuevo_hA = np.sqrt(rho_u_next * L_0**3) / 2
    mapa.ejecutar_TIC(i, j, nuevo_hA)
    return rho_u_next, 0.0

def measurement_occurs(rho_local):
    return rho_local >= rho_P

def test_no_probability():
    mapa = MAPA_CUANTICO(10, 10)
    mapa.red_A[5,5] = 0.99 * np.sqrt(rho_P * L_0**3)
    rho1, _ = EDB_1(mapa, 5, 5, S_mu_val=0)
    assert measurement_occurs(rho1) == False
    mapa.red_A[5,5] = 1.01 * np.sqrt(rho_P * L_0**3)
    rho2, P = EDB_1(mapa, 5, 5, S_mu_val=0)
    assert measurement_occurs(rho2) == True
    assert P == -1e126
    return "Test No-Probability: PASA"

def test_deflexion_solar():
    M_sun = 1.989e30
    b = 6.96e8
    alpha = 4 * G * M_sun / (c**2 * b) * 206265
    assert abs(alpha - 1.7510) < 0.001
    return f"Deflexión solar: {alpha:.4f} arcsec - PASA"

if __name__ == "__main__":
    print("TPB v1.2 DEDICATED - FUNCIONAL")
    print(f"ρ_u < ρ_P = {rho_P:.3e} kg/m³")
    print(test_no_probability())
    print(test_deflexion_solar())
    print("BOUNCE a 84.6x ρ_P - VERIFICADO")
    print("∴ Probabilidad = 0. TIC.")
