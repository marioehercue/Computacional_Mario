import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LEER PARÁMETROS
# ============================================================

def leer_parametros(nombre_archivo):
    parametros = {}

    with open(nombre_archivo, "r", encoding="utf-8") as f:
        for linea in f:
            clave, valor = linea.split(maxsplit=1)
            parametros[clave] = valor.strip()

    parametros["N"] = int(parametros["N"])
    parametros["P"] = int(parametros["P"])
    parametros["T"] = float(parametros["T"])

    return parametros


# ============================================================
# VISUALIZACIÓN
# ============================================================

def mostrar_varios_patrones(patrones, titulos=None):
    patrones = np.array(patrones)
    n = len(patrones)

    if titulos is None:
        titulos = [f"Patrón {i+1}" for i in range(n)]

    plt.figure(figsize=(4 * n, 4))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(patrones[i], cmap="gray_r", interpolation="nearest")
        plt.title(titulos[i])
        plt.axis("off")

    plt.show()


def representar_energia(energias):
    plt.figure(figsize=(6, 4))
    plt.plot(energias)
    plt.xlabel("Paso Monte Carlo")
    plt.ylabel("Energía")
    plt.title("Evolución de la energía")
    plt.grid(True)
    plt.show()


def representar_solapamientos(solapamientos):
    tiempos = np.arange(solapamientos.shape[0])
    num_patrones = solapamientos.shape[1]

    plt.figure(figsize=(7, 4))

    for mu in range(num_patrones):
        plt.plot(tiempos, solapamientos[:, mu], label=f"Patrón {mu+1}")

    plt.xlabel("Paso Monte Carlo")
    plt.ylabel("Solapamiento")
    plt.title("Solapamiento con los patrones almacenados")
    plt.ylim(-1.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    parametros = leer_parametros("datos/parametros.dat")

    N = parametros["N"]
    P = parametros["P"]

    patrones = np.loadtxt("datos/patrones.dat").reshape(P, N, N)
    estado_inicial = np.loadtxt("datos/estado_inicial.dat")
    estado_final = np.loadtxt("datos/estado_final.dat")
    energias = np.loadtxt("datos/energias.dat")
    solapamientos = np.loadtxt("datos/solapamientos.dat")

    mostrar_varios_patrones(
        [patrones[0], estado_inicial, estado_final],
        ["Patrón objetivo", "Estado inicial", "Estado final"]
    )

    representar_energia(energias)
    representar_solapamientos(solapamientos)