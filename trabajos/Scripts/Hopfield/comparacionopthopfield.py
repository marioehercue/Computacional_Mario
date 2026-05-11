import numpy as np
import time
from numba import njit


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def crear_rng(seed=None):
    return np.random.default_rng(seed)


def crear_patron_aleatorio(N, prob_activa=0.3, rng=None):
    if rng is None:
        rng = crear_rng()

    patron = rng.random((N, N)) < prob_activa
    return patron.astype(np.int64)


def deformar_patron(patron, fraccion_ruido=0.2, rng=None):
    if rng is None:
        rng = crear_rng()

    patron_deformado = patron.copy()
    N = patron.shape[0]

    numero_neuronas = N * N
    numero_cambios = int(fraccion_ruido * numero_neuronas)

    indices = rng.choice(numero_neuronas, size=numero_cambios, replace=False)

    patron_plano = patron_deformado.ravel()
    patron_plano[indices] = 1 - patron_plano[indices]

    return patron_deformado


def calcular_bias(patrones):
    return np.mean(patrones)


def calcular_pesos(patrones):
    patrones = np.array(patrones)

    P, N, _ = patrones.shape
    M = N * N

    a = calcular_bias(patrones)

    if a == 0 or a == 1:
        raise ValueError("El bias no puede ser 0 ni 1.")

    patrones_planos = patrones.reshape(P, M)
    patrones_centrados = patrones_planos - a

    pesos = (patrones_centrados.T @ patrones_centrados) / (a * (1 - a) * M)

    np.fill_diagonal(pesos, 0)

    return pesos, a


def calcular_umbrales(pesos):
    return 0.5 * np.sum(pesos, axis=1)


# ============================================================
# VERSIÓN PYTHON NORMAL
# ============================================================

def calcular_energia_python(estado_plano, pesos, umbrales):
    energia_interaccion = -0.5 * estado_plano @ pesos @ estado_plano
    energia_umbral = np.sum(umbrales * estado_plano)

    return energia_interaccion + energia_umbral


def evolucionar_python(estado_inicial, pesos, umbrales, T, num_pasos_mc, indices_random, numeros_random):
    """
    Evolución de Hopfield usando Python normal.

    indices_random:
        índices de neuronas que se intentan cambiar.

    numeros_random:
        números aleatorios entre 0 y 1 para aplicar Metropolis.
    """

    estado = estado_inicial.copy()
    estado_plano = estado.ravel()

    M = estado_plano.size

    energias = np.zeros(num_pasos_mc + 1)
    aceptaciones = np.zeros(num_pasos_mc + 1)

    energias[0] = calcular_energia_python(estado_plano, pesos, umbrales)
    aceptaciones[0] = 0.0

    contador_random = 0

    for paso in range(1, num_pasos_mc + 1):

        aceptados = 0

        for _ in range(M):

            indice = indices_random[contador_random]
            numero_aleatorio = numeros_random[contador_random]
            contador_random += 1

            valor_actual = estado_plano[indice]
            valor_nuevo = 1 - valor_actual

            delta_s = valor_nuevo - valor_actual

            campo = np.dot(pesos[indice], estado_plano)

            delta_H = -delta_s * campo + umbrales[indice] * delta_s

            if delta_H <= 0:
                aceptar = True
            else:
                probabilidad = np.exp(-delta_H / T)
                aceptar = numero_aleatorio < probabilidad

            if aceptar:
                estado_plano[indice] = valor_nuevo
                aceptados += 1

        energias[paso] = calcular_energia_python(estado_plano, pesos, umbrales)
        aceptaciones[paso] = aceptados / M

    return estado, energias, aceptaciones


# ============================================================
# VERSIÓN NUMBA
# ============================================================

@njit
def calcular_energia_numba(estado_plano, pesos, umbrales):
    M = estado_plano.size

    energia_interaccion = 0.0
    energia_umbral = 0.0

    for i in range(M):
        energia_umbral += umbrales[i] * estado_plano[i]

        for j in range(M):
            energia_interaccion += pesos[i, j] * estado_plano[i] * estado_plano[j]

    energia_interaccion *= -0.5

    return energia_interaccion + energia_umbral


@njit
def evolucionar_numba(estado_inicial, pesos, umbrales, T, num_pasos_mc, indices_random, numeros_random):
    """
    Misma evolución que evolucionar_python, pero compilada con Numba.
    """

    estado = estado_inicial.copy()
    estado_plano = estado.ravel()

    M = estado_plano.size

    energias = np.zeros(num_pasos_mc + 1)
    aceptaciones = np.zeros(num_pasos_mc + 1)

    energias[0] = calcular_energia_numba(estado_plano, pesos, umbrales)
    aceptaciones[0] = 0.0

    contador_random = 0

    for paso in range(1, num_pasos_mc + 1):

        aceptados = 0

        for _ in range(M):

            indice = indices_random[contador_random]
            numero_aleatorio = numeros_random[contador_random]
            contador_random += 1

            valor_actual = estado_plano[indice]
            valor_nuevo = 1 - valor_actual

            delta_s = valor_nuevo - valor_actual

            campo = 0.0
            for j in range(M):
                campo += pesos[indice, j] * estado_plano[j]

            delta_H = -delta_s * campo + umbrales[indice] * delta_s

            if delta_H <= 0:
                aceptar = True
            else:
                probabilidad = np.exp(-delta_H / T)
                aceptar = numero_aleatorio < probabilidad

            if aceptar:
                estado_plano[indice] = valor_nuevo
                aceptados += 1

        energias[paso] = calcular_energia_numba(estado_plano, pesos, umbrales)
        aceptaciones[paso] = aceptados / M

    return estado, energias, aceptaciones


# ============================================================
# MAIN: COMPARACIÓN
# ============================================================

if __name__ == "__main__":

    rng = crear_rng(seed=123)

    # ==========================
    # PARÁMETROS
    # ==========================

    N = 40
    P = 2
    prob_activa = 0.25
    T = 1e-4
    num_pasos_mc = 50
    fraccion_ruido = 0.2

    M = N * N

    print("\n========== COMPARACIÓN HOPFIELD ==========")
    print(f"N = {N}")
    print(f"P = {P}")
    print(f"M = {M}")
    print(f"T = {T}")
    print(f"Pasos Monte Carlo = {num_pasos_mc}")

    # ==========================
    # PATRONES
    # ==========================

    patrones = np.array([
    crear_patron_aleatorio(N, prob_activa, rng)
    for _ in range(P)
    ], dtype=np.float64)

    patron_objetivo = patrones[0]

    # ==========================
    # PESOS Y UMBRALES
    # ==========================

    pesos, a = calcular_pesos(patrones)
    umbrales = calcular_umbrales(pesos)

    pesos = pesos.astype(np.float64)
    umbrales = umbrales.astype(np.float64)

    # ==========================
    # ESTADO INICIAL
    # ==========================

    estado_inicial = deformar_patron(
    patron_objetivo,
    fraccion_ruido=fraccion_ruido,
    rng=rng
    ).astype(np.float64)

    # ==========================
    # NÚMEROS ALEATORIOS PREGENERADOS
    # ==========================

    total_intentos = num_pasos_mc * M

    indices_random = rng.integers(0, M, size=total_intentos).astype(np.int64)
    numeros_random = rng.random(total_intentos).astype(np.float64)

    # ==========================
    # PYTHON NORMAL
    # ==========================

    t0 = time.perf_counter()

    estado_py, energias_py, aceptaciones_py = evolucionar_python(
        estado_inicial,
        pesos,
        umbrales,
        T,
        num_pasos_mc,
        indices_random,
        numeros_random
    )

    t1 = time.perf_counter()

    tiempo_python = t1 - t0

    # ==========================
    # NUMBA
    # ==========================

    # Primera llamada: compila. No se mide.
    evolucionar_numba(
        estado_inicial,
        pesos,
        umbrales,
        T,
        num_pasos_mc,
        indices_random,
        numeros_random
    )

    # Segunda llamada: ejecución real medida.
    t0 = time.perf_counter()

    estado_nb, energias_nb, aceptaciones_nb = evolucionar_numba(
        estado_inicial,
        pesos,
        umbrales,
        T,
        num_pasos_mc,
        indices_random,
        numeros_random
    )

    t1 = time.perf_counter()

    tiempo_numba = t1 - t0

    # ==========================
    # COMPROBACIONES
    # ==========================

    misma_energia_final = np.isclose(energias_py[-1], energias_nb[-1])
    mismo_estado_final = np.array_equal(estado_py, estado_nb)

    speedup = tiempo_python / tiempo_numba

    # ==========================
    # RESULTADOS
    # ==========================

    print("\n========== RESULTADOS ==========")

    print(f"Energía final Python = {energias_py[-1]:.6f}")
    print(f"Energía final Numba  = {energias_nb[-1]:.6f}")

    print(f"\n¿Misma energía final? {misma_energia_final}")
    print(f"¿Mismo estado final?  {mismo_estado_final}")

    print("\n========== TIEMPOS ==========")
    print(f"Tiempo Python normal = {tiempo_python:.6f} s")
    print(f"Tiempo Numba         = {tiempo_numba:.6f} s")
    print(f"Speed-up             = {speedup:.2f}")

    print("\nInterpretación:")
    print(f"La versión con Numba es aproximadamente {speedup:.2f} veces más rápida.")