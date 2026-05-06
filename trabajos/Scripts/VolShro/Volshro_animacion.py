import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. ARCHIVO DE ENTRADA
# ============================================================

archivo_datos = "datos.dat"


# ============================================================
# 2. LECTURA DE DATOS
# ============================================================

def leer_datos(nombre_archivo):
    """
    Lee el archivo datos.dat generado por Volshro_simulacion.py.

    El formato esperado es:

        j  V_tilde  rho_t0  rho_t5  rho_t10  ...

    donde rho = |phi|^2.
    """

    datos = np.loadtxt(nombre_archivo)

    j = datos[:, 0]
    V_tilde = datos[:, 1]

    densidades = datos[:, 2:]

    return j, V_tilde, densidades


def leer_tiempos_desde_cabecera(nombre_archivo):
    """
    Intenta leer los tiempos guardados desde la cabecera de datos.dat.

    Busca columnas llamadas rho_t0, rho_t5, rho_t10, etc.
    Si no las encuentra, devuelve simplemente 0, 1, 2, ...
    """

    with open(nombre_archivo, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    for linea in lineas:
        if "rho_t" in linea:
            partes = linea.replace("#", "").split()

            tiempos = []

            for palabra in partes:
                if palabra.startswith("rho_t"):
                    tiempo = int(palabra.replace("rho_t", ""))
                    tiempos.append(tiempo)

            if len(tiempos) > 0:
                return np.array(tiempos)

    return None


# ============================================================
# 3. ANIMACIÓN
# ============================================================

def animar_densidad(j, V_tilde, densidades, tiempos=None):
    """
    Anima la densidad de probabilidad |phi|^2 junto con el potencial.

    Como V_tilde puede tener otra escala, se reescala solo para visualizarlo
    en la misma gráfica que la densidad.
    """

    if tiempos is None:
        tiempos = np.arange(densidades.shape[1])

    rho_max = np.max(densidades)

    if np.max(V_tilde) > 0:
        V_grafica = V_tilde / np.max(V_tilde) * rho_max * 0.8
    else:
        V_grafica = V_tilde

    fig, ax = plt.subplots(figsize=(9, 5))

    linea_rho, = ax.plot(j, densidades[:, 0], lw=2, label=r"$|\Phi(j,t)|^2$")
    linea_V, = ax.plot(j, V_grafica, "--", lw=2, label="Potencial reescalado")

    ax.set_xlabel("j")
    ax.set_ylabel(r"Densidad de probabilidad $|\Phi|^2$")
    ax.set_title("Evolución del paquete de ondas")
    ax.grid(True, alpha=0.4)
    ax.legend()

    ax.set_xlim(j[0], j[-1])
    ax.set_ylim(0, rho_max * 1.15)

    texto_tiempo = ax.text(
        0.02,
        0.92,
        "",
        transform=ax.transAxes,
        fontsize=11
    )

    def actualizar(frame):
        linea_rho.set_ydata(densidades[:, frame])
        texto_tiempo.set_text(f"Paso temporal: {tiempos[frame]}")
        return linea_rho, texto_tiempo

    animacion = FuncAnimation(
        fig,
        actualizar,
        frames=densidades.shape[1],
        interval=40,
        blit=True
    )

    plt.show()

    return animacion


# ============================================================
# 4. PROGRAMA PRINCIPAL
# ============================================================

j, V_tilde, densidades = leer_datos(archivo_datos)

tiempos = leer_tiempos_desde_cabecera(archivo_datos)

print("Archivo leído correctamente")
print("---------------------------")
print(f"Número de puntos espaciales       = {len(j)}")
print(f"Número de instantes guardados     = {densidades.shape[1]}")
print(f"Máximo de densidad                = {np.max(densidades):.6e}")
print(f"Máximo del potencial reescalado   = {np.max(V_tilde):.6e}")
print()

animacion = animar_densidad(j, V_tilde, densidades, tiempos)