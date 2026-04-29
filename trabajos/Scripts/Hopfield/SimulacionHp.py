import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. GENERADOR DE NÚMEROS ALEATORIOS
# ============================================================

def crear_rng(seed=None):
    """
    Crea un generador moderno de números aleatorios de NumPy.

    Usamos np.random.default_rng(), que es la forma recomendada
    actualmente frente a np.random.seed().
    """
    return np.random.default_rng(seed)


# ============================================================
# 2. PATRONES
# ============================================================

def crear_patron_aleatorio(N, prob_activa=0.3, rng=None):
    """
    Crea un patrón aleatorio N x N con valores 0 y 1.

    prob_activa:
        Probabilidad de que una neurona esté activa, es decir, valga 1.
    """
    if rng is None:
        rng = crear_rng()

    patron = rng.random((N, N)) < prob_activa
    return patron.astype(int)


def deformar_patron(patron, fraccion_ruido=0.2, rng=None):
    """
    Deforma un patrón cambiando una fracción de sus neuronas.

    Si una neurona vale 1 pasa a 0.
    Si una neurona vale 0 pasa a 1.
    """
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
    """
    Calcula el bias medio a.

    patrones debe ser un array de forma:
        (P, N, N)

    donde:
        P = número de patrones almacenados.
    """
    return np.mean(patrones)


# ============================================================
# 3. PESOS SINÁPTICOS
# ============================================================

def calcular_pesos(patrones):
    """
    Calcula la matriz de pesos de la red de Hopfield modificada.

    Los patrones tienen valores 0 y 1.

    Fórmula usada:

        w_ij = 1 / [a(1-a) M] * sum_mu (xi_i^mu - a)(xi_j^mu - a)

    donde:
        M = N*N = número total de neuronas.
        a = actividad media o bias.
    """
    patrones = np.array(patrones)

    P, N, _ = patrones.shape
    M = N * N

    a = calcular_bias(patrones)

    if a == 0 or a == 1:
        raise ValueError("El bias no puede ser 0 ni 1, porque aparece en el denominador.")

    patrones_planos = patrones.reshape(P, M)

    patrones_centrados = patrones_planos - a

    pesos = (patrones_centrados.T @ patrones_centrados) / (a * (1 - a) * M)

    # No permitimos autoconexiones
    np.fill_diagonal(pesos, 0)

    return pesos, a


# ============================================================
# 4. ENERGÍA Y DINÁMICA DE METROPOLIS
# ============================================================

def calcular_umbrales(pesos):
    """
    Calcula los umbrales theta_i.

    Según el enunciado:

        theta_i = 1/2 * sum_j w_ij
    """
    return 0.5 * np.sum(pesos, axis=1)


def calcular_energia(estado, pesos, umbrales):
    """
    Calcula la energía de una configuración.

    H = -1/2 sum_ij w_ij s_i s_j + sum_i theta_i s_i

    estado:
        matriz N x N con valores 0 y 1.
    """
    s = estado.ravel()

    energia_interaccion = -0.5 * s @ pesos @ s
    energia_umbral = np.sum(umbrales * s)

    return energia_interaccion + energia_umbral


def calcular_delta_energia(estado, indice, pesos, umbrales):
    """
    Calcula Delta H al cambiar una única neurona.

    indice:
        índice plano de la neurona que se intenta cambiar.

    Si s_i = 0, se prueba 0 -> 1.
    Si s_i = 1, se prueba 1 -> 0.
    """
    s = estado.ravel()

    valor_actual = s[indice]
    valor_nuevo = 1 - valor_actual

    delta_s = valor_nuevo - valor_actual

    campo = np.dot(pesos[indice], s)

    delta_H = -delta_s * campo + umbrales[indice] * delta_s

    return delta_H


def paso_metropolis(estado, pesos, umbrales, T, rng=None):
    """
    Realiza un intento de cambio de una neurona usando Metropolis.

    Devuelve:
        True si el cambio se acepta.
        False si se rechaza.
    """
    if rng is None:
        rng = crear_rng()

    N = estado.shape[0]
    M = N * N

    indice = rng.integers(0, M)

    delta_H = calcular_delta_energia(estado, indice, pesos, umbrales)

    if delta_H <= 0:
        aceptar = True
    else:
        probabilidad = np.exp(-delta_H / T)
        aceptar = rng.random() < probabilidad

    if aceptar:
        estado.ravel()[indice] = 1 - estado.ravel()[indice]

    return aceptar


def paso_montecarlo(estado, pesos, umbrales, T, rng=None):
    """
    Realiza un paso Monte Carlo completo.

    Un paso Monte Carlo equivale a N*N intentos de cambio,
    de forma que, en promedio, cada neurona intenta cambiar una vez.
    """
    if rng is None:
        rng = crear_rng()

    N = estado.shape[0]
    M = N * N

    aceptados = 0

    for _ in range(M):
        if paso_metropolis(estado, pesos, umbrales, T, rng):
            aceptados += 1

    fraccion_aceptados = aceptados / M

    return fraccion_aceptados


def evolucionar_red(estado_inicial, pesos, umbrales, T, num_pasos_mc, rng=None):
    """
    Evoluciona la red durante num_pasos_mc pasos Monte Carlo.

    Devuelve:
        estado_final
        energias
        aceptaciones
    """
    if rng is None:
        rng = crear_rng()

    estado = estado_inicial.copy()

    energias = []
    aceptaciones = []

    for _ in range(num_pasos_mc):
        fraccion_aceptados = paso_montecarlo(estado, pesos, umbrales, T, rng)

        energia = calcular_energia(estado, pesos, umbrales)

        energias.append(energia)
        aceptaciones.append(fraccion_aceptados)

    return estado, np.array(energias), np.array(aceptaciones)


# ============================================================
# 5. SOLAPAMIENTO
# ============================================================

def calcular_solapamiento(estado, patron):
    """
    Calcula el solapamiento entre el estado actual y un patrón.

    Fórmula:

        m = 1 / [M a(1-a)] * sum_i (xi_i - a)(s_i - a)

    donde:
        M = N*N
        a = actividad media del patrón
    """
    s = estado.ravel()
    xi = patron.ravel()

    M = len(xi)
    a = np.mean(xi)

    if a == 0 or a == 1:
        raise ValueError("El patrón no puede tener bias 0 ni 1.")

    m = np.sum((xi - a) * (s - a)) / (M * a * (1 - a))

    return m


def calcular_solapamientos(estado, patrones):
    """
    Calcula el solapamiento del estado con todos los patrones almacenados.

    Devuelve un array:
        [m_1, m_2, ..., m_P]
    """
    return np.array([calcular_solapamiento(estado, patron) for patron in patrones])


def evolucionar_red_con_solapamiento(
    estado_inicial, pesos, umbrales, patrones, T, num_pasos_mc, rng=None
):
    """
    Evoluciona la red y guarda:
        - energía
        - fracción de cambios aceptados
        - solapamiento con cada patrón
    """
    if rng is None:
        rng = crear_rng()

    estado = estado_inicial.copy()

    energias = []
    aceptaciones = []
    solapamientos = []

    # Medimos también el estado inicial
    energias.append(calcular_energia(estado, pesos, umbrales))
    aceptaciones.append(0)
    solapamientos.append(calcular_solapamientos(estado, patrones))

    for _ in range(num_pasos_mc):
        fraccion_aceptados = paso_montecarlo(estado, pesos, umbrales, T, rng)

        energias.append(calcular_energia(estado, pesos, umbrales))
        aceptaciones.append(fraccion_aceptados)
        solapamientos.append(calcular_solapamientos(estado, patrones))

    return estado, np.array(energias), np.array(aceptaciones), np.array(solapamientos)


def representar_solapamientos(solapamientos):
    """
    Representa el solapamiento con cada patrón en función del tiempo.
    """
    num_pasos = solapamientos.shape[0]
    num_patrones = solapamientos.shape[1]

    tiempos = np.arange(num_pasos)

    plt.figure(figsize=(7, 4))

    for mu in range(num_patrones):
        plt.plot(tiempos, solapamientos[:, mu], marker="o", label=f"Patrón {mu+1}")

    plt.xlabel("Paso Monte Carlo")
    plt.ylabel("Solapamiento")
    plt.title("Solapamiento con los patrones almacenados")
    plt.ylim(-1.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# 6. ESTUDIO EN TEMPERATURA
# ============================================================

def estudiar_temperaturas(
    patron_objetivo,
    patrones,
    pesos,
    umbrales,
    temperaturas,
    fraccion_ruido=0.2,
    num_pasos_mc=20,
    num_repeticiones=5,
    rng=None
):
    """
    Estudia cómo cambia la recuperación de un patrón con la temperatura.

    Para cada temperatura:
        - se deforma el patrón objetivo
        - se evoluciona la red
        - se mide el solapamiento final con el patrón objetivo

    Se repite varias veces para reducir el efecto del azar.
    """
    if rng is None:
        rng = crear_rng()

    solapamientos_medios = []
    solapamientos_std = []

    for T in temperaturas:
        solapamientos_T = []

        for _ in range(num_repeticiones):
            estado_inicial = deformar_patron(
                patron_objetivo,
                fraccion_ruido=fraccion_ruido,
                rng=rng
            )

            estado_final, _, _, solapamientos = evolucionar_red_con_solapamiento(
                estado_inicial=estado_inicial,
                pesos=pesos,
                umbrales=umbrales,
                patrones=patrones,
                T=T,
                num_pasos_mc=num_pasos_mc,
                rng=rng
            )

            # Solapamiento final con el patrón objetivo, que suponemos patrón 1
            solapamiento_final = solapamientos[-1, 0]
            solapamientos_T.append(solapamiento_final)

        solapamientos_medios.append(np.mean(solapamientos_T))
        solapamientos_std.append(np.std(solapamientos_T))

    return np.array(solapamientos_medios), np.array(solapamientos_std)


def representar_solapamiento_temperatura(temperaturas, solapamientos_medios, solapamientos_std):
    """
    Representa el solapamiento final medio frente a la temperatura.
    """
    plt.figure(figsize=(7, 4))

    plt.errorbar(
        temperaturas,
        solapamientos_medios,
        yerr=solapamientos_std,
        marker="o",
        capsize=4
    )

    plt.xlabel("Temperatura")
    plt.ylabel("Solapamiento final medio")
    plt.title("Recuperación de memoria en función de la temperatura")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()

# ============================================================
# 7. EXPERIMENTO CONTROLADO (N, P, T FIJOS)
# ============================================================

def experimento_simple(
    N=20,
    P=5,
    prob_activa=0.25,
    T=1e-4,
    num_pasos_mc=30,
    fraccion_ruido=0.2,
    rng=None
):
    """
    Ejecuta un experimento completo de la red de Hopfield con:

        - N: tamaño de la red (NxN)
        - P: número de patrones
        - T: temperatura

    Devuelve resultados y muestra gráficas.
    """

    if rng is None:
        rng = crear_rng()

    # =====================================================
    # 1. GENERAR PATRONES
    # =====================================================
    patrones = np.array([
        crear_patron_aleatorio(N, prob_activa, rng)
        for _ in range(P)
    ])

    patron_objetivo = patrones[0]

    # =====================================================
    # 2. PESOS Y UMBRALES
    # =====================================================
    pesos, a = calcular_pesos(patrones)
    umbrales = calcular_umbrales(pesos)

    # =====================================================
    # 3. ESTADO INICIAL (DEFORMADO)
    # =====================================================
    modo_inicial = "aleatorio"  # "aleatorio" o "deformado"
    if modo_inicial == "deformado":
        estado_inicial = deformar_patron(
            patron_objetivo,
            fraccion_ruido=fraccion_ruido,
            rng=rng
        )
    else:
        estado_inicial = crear_patron_aleatorio(N, prob_activa, rng)

    # =====================================================
    # 4. EVOLUCIÓN
    # =====================================================
    estado_final, energias, aceptaciones, solapamientos = evolucionar_red_con_solapamiento(
        estado_inicial,
        pesos,
        umbrales,
        patrones,
        T,
        num_pasos_mc,
        rng
    )

    # =====================================================
    # 5. RESULTADOS
    # =====================================================
    print("\n========== RESULTADOS ==========")
    print(f"N = {N}  |  P = {P}  |  T = {T}")
    print(f"Bias medio a = {a:.3f}")
    print(f"Energía inicial = {energias[0]:.3f}")
    print(f"Energía final   = {energias[-1]:.3f}")
    print(f"Fracción media aceptada = {np.mean(aceptaciones[1:]):.3f}")

    print("\nSolapamientos finales:")
    for i in range(P):
        print(f"m_{i+1} = {solapamientos[-1, i]:.3f}")

# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":

    import os

    os.makedirs("datos", exist_ok=True)

    rng = crear_rng(seed=123)

    # ==========================
    # PARÁMETROS
    # ==========================

    N = 20
    P = 2
    prob_activa = 0.25
    T = 1e-4
    num_pasos_mc = 30
    fraccion_ruido = 0.2
    modo_inicial = "deformado"   # "deformado" o "aleatorio"

    # ==========================
    # PATRONES
    # ==========================

    patrones = np.array([
        crear_patron_aleatorio(N, prob_activa, rng)
        for _ in range(P)
    ])

    patron_objetivo = patrones[0]

    # ==========================
    # PESOS Y UMBRALES
    # ==========================

    pesos, a = calcular_pesos(patrones)
    umbrales = calcular_umbrales(pesos)

    # ==========================
    # ESTADO INICIAL
    # ==========================

    if modo_inicial == "deformado":
        estado_inicial = deformar_patron(
            patron_objetivo,
            fraccion_ruido=fraccion_ruido,
            rng=rng
        )

    elif modo_inicial == "aleatorio":
        estado_inicial = crear_patron_aleatorio(
            N,
            prob_activa=prob_activa,
            rng=rng
        )

    else:
        raise ValueError("modo_inicial debe ser 'deformado' o 'aleatorio'.")

    # ==========================
    # EVOLUCIÓN
    # ==========================

    estado_final, energias, aceptaciones, solapamientos = evolucionar_red_con_solapamiento(
        estado_inicial=estado_inicial,
        pesos=pesos,
        umbrales=umbrales,
        patrones=patrones,
        T=T,
        num_pasos_mc=num_pasos_mc,
        rng=rng
    )

    # ==========================
    # GUARDAR DATOS .dat
    # ==========================

    np.savetxt("datos/patrones.dat", patrones.reshape(P, N * N), fmt="%d")
    np.savetxt("datos/estado_inicial.dat", estado_inicial, fmt="%d")
    np.savetxt("datos/estado_final.dat", estado_final, fmt="%d")
    np.savetxt("datos/energias.dat", energias)
    np.savetxt("datos/aceptaciones.dat", aceptaciones)
    np.savetxt("datos/solapamientos.dat", solapamientos)

    with open("datos/parametros.dat", "w", encoding="utf-8") as f:
        f.write(f"N {N}\n")
        f.write(f"P {P}\n")
        f.write(f"prob_activa {prob_activa}\n")
        f.write(f"T {T}\n")
        f.write(f"num_pasos_mc {num_pasos_mc}\n")
        f.write(f"fraccion_ruido {fraccion_ruido}\n")
        f.write(f"modo_inicial {modo_inicial}\n")
        f.write(f"bias {a}\n")

    # ==========================
    # RESUMEN POR TERMINAL
    # ==========================

    print("\n========== SIMULACIÓN COMPLETADA ==========")
    print(f"N = {N}")
    print(f"P = {P}")
    print(f"T = {T}")
    print(f"Modo inicial = {modo_inicial}")
    print(f"Bias medio a = {a:.3f}")
    print(f"Energía inicial = {energias[0]:.3f}")
    print(f"Energía final   = {energias[-1]:.3f}")
    print(f"Fracción media aceptada = {np.mean(aceptaciones[1:]):.3f}")

    print("\nSolapamientos finales:")
    for i in range(P):
        print(f"m_{i+1} = {solapamientos[-1, i]:.3f}")

    print("\nDatos guardados en la carpeta datos/")