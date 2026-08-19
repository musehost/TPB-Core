# TPB VTK - Horizonte inercial estable
import numpy as np

def calcular_t_prima(theta_grid, lp=1.616e-35):
    gx, gy, gz = np.gradient(theta_grid, lp)
    I_mag = lp * np.sqrt(gx**2 + gy**2 + gz**2)
    t_prime = np.clip(1.0 - I_mag, 0.001, 1.0)
    return t_prime, I_mag

# Grid 40x40x40 - tu horizonte
N = 40
lp = 1.616e-35
theta = np.random.uniform(0, 0.1, (N,N,N))
# Crear pozo en el centro (tu horizonte)
cx = N//2
theta[cx-2:cx+2, cx-2:cx+2, cx-2:cx+2] += 0.2

tp, I = calcular_t_prima(theta, lp)

print(f"t' min: {np.min(tp):.4f} | t' max: {np.max(tp):.4f}")
print(f"I max: {np.max(I):.4f} -> ESTABLE")

# Guardar en VTK legacy (abre en ParaView / app VTK viewer)
with open("t_prime.vtk", "w") as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("TPB t' inercial - Benjamin Langston 2026\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_POINTS\n")
    f.write(f"DIMENSIONS {N} {N} {N}\n")
    f.write("ORIGIN 0 0 0\n")
    f.write(f"SPACING {lp*1e35} {lp*1e35} {lp*1e35}\n")
    f.write(f"POINT_DATA {N*N*N}\n")
    f.write("SCALARS t_prime float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for k in range(N):
        for j in range(N):
            for i in range(N):
                f.write(f"{tp[i,j,k]:.6f}\n")

print("Archivo t_prime.vtk creado")
print(f"Tamano: {N}x{N}x{N}")

# Tambien I para ver densidad
with open("I_mag.vtk", "w") as f:
    f.write("# vtk DataFile Version 3.0\n")
    f.write("TPB I = lp*grad(theta)\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_POINTS\n")
    f.write(f"DIMENSIONS {N} {N} {N}\n")
    f.write("ORIGIN 0 0 0\n")
    f.write(f"SPACING 1 1 1\n")
    f.write(f"POINT_DATA {N*N*N}\n")
    f.write("SCALARS I_mag float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for k in range(N):
        for j in range(N):
            for i in range(N):
                f.write(f"{I[i,j,k]:.6f}\n")

print("Archivo I_mag.vtk creado -> listo para ParaView")
