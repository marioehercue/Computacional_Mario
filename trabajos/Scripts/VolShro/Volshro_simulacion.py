import numpy as np


# ============================================================
# 1. PARÁMETROS PRINCIPALES
# ============================================================

N = 500
n_ciclos = N // 4
lam = 0.3

n_pasos = 1500
guardar_cada = 5

m = 1000
seed = 12345

archivo_salida = "datos.dat"
archivo_normas = "normas.dat"
archivo_detectores = "detectores.dat"
archivo_nD = "nD.dat"
archivo_K = "coeficiente_transmision.dat"
archivo_observables = "observables.dat"


# ============================================================
# 2. FUNCIONES BÁSICAS
# ============================================================

def crear_parametros(N, n_ciclos):
    """
    Calcula los parámetros discretos principales del problema.

    N:
        Número de intervalos espaciales. La red tendrá N+1 puntos: j = 0, ..., N.

    n_ciclos:
        Número de oscilaciones completas de la onda plana en la red.

    Devuelve:
        j: array con los índices espaciales.
        k_tilde: número de onda discreto.
        s_tilde: paso temporal reescalado.
    """

    if n_ciclos < 1:
        raise ValueError("n_ciclos debe ser al menos 1.")

    if n_ciclos > N // 4:
        raise ValueError("n_ciclos debe cumplir n_ciclos <= N/4.")

    j = np.arange(N + 1)

    k_tilde = 2 * np.pi * n_ciclos / N

    s_tilde = 1 / (4 * k_tilde**2)

    return j, k_tilde, s_tilde


def crear_potencial_cuadrado(N, k_tilde, lam):
    """
    Crea el potencial cuadrado reescalado V_tilde.

    La barrera está entre 2N/5 y 3N/5.
    Su altura es lam * k_tilde^2.
    """

    V_tilde = np.zeros(N + 1)

    inicio = 2 * N // 5
    fin = 3 * N // 5

    V_tilde[inicio:fin + 1] = lam * k_tilde**2

    return V_tilde


def crear_onda_inicial(N, k_tilde):
    """
    Crea la función de onda inicial:

        phi_j = exp(i k_tilde j) * exp[-8(4j - N)^2 / N^2]

    Es un paquete gaussiano localizado a la izquierda,
    con una fase oscilante que le da momento hacia la derecha.
    """

    j = np.arange(N + 1)

    fase = np.exp(1j * k_tilde * j)

    envolvente = np.exp(-8 * (4 * j - N)**2 / N**2)

    phi = fase * envolvente

    # Condiciones de contorno: paredes infinitas en j = 0 y j = N
    phi[0] = 0.0
    phi[N] = 0.0

    return phi


def norma(phi):
    """
    Calcula la norma discreta de la función de onda.

    En este planteamiento usamos la suma discreta:
        sum_j |phi_j|^2
    """

    return np.sum(np.abs(phi)**2)


def normalizar(phi):
    """
    Normaliza la función de onda para que su norma sea 1.
    """

    n = norma(phi)

    if n == 0:
        raise ValueError("No se puede normalizar una función de onda con norma cero.")

    return phi / np.sqrt(n)


# ============================================================
# 3. MÉTODO DE CAYLEY
# ============================================================

def calcular_alpha(N, s_tilde, V_tilde):
    """
    Calcula los coeficientes alpha de la recurrencia tridiagonal.

    Estos coeficientes no dependen del tiempo, por lo que se calculan
    una sola vez al inicio de la simulación.
    """

    alpha = np.zeros(N + 1, dtype=complex)

    # Condición final: alpha_{N-1} = 0
    alpha[N - 1] = 0.0 + 0.0j

    # Recorremos hacia atrás: j = N-1, ..., 1
    for j in range(N - 1, 0, -1):

        A0 = -2.0 + 2.0j / s_tilde - V_tilde[j]

        gamma = 1.0 / (A0 + alpha[j])

        alpha[j - 1] = -gamma

    return alpha


def calcular_beta(phi, alpha, N, s_tilde, V_tilde):
    """
    Calcula los coeficientes beta para el estado phi actual.

    A diferencia de alpha, beta sí depende de phi, así que hay que
    recalcularlo en cada paso temporal.
    """

    beta = np.zeros(N + 1, dtype=complex)

    # Condición final: beta_{N-1} = 0
    beta[N - 1] = 0.0 + 0.0j

    # Recorremos hacia atrás: j = N-1, ..., 1
    for j in range(N - 1, 0, -1):

        A0 = -2.0 + 2.0j / s_tilde - V_tilde[j]

        b = 4.0j * phi[j] / s_tilde

        gamma = 1.0 / (A0 + alpha[j])

        beta[j - 1] = gamma * (b - beta[j])

    return beta


def calcular_chi(alpha, beta, N):
    """
    Calcula chi usando la recurrencia:

        chi_{j+1} = alpha_j chi_j + beta_j

    con la condición de contorno chi_0 = 0.
    """

    chi = np.zeros(N + 1, dtype=complex)

    chi[0] = 0.0 + 0.0j

    for j in range(0, N):
        chi[j + 1] = alpha[j] * chi[j] + beta[j]

    # Condición de contorno
    chi[0] = 0.0 + 0.0j
    chi[N] = 0.0 + 0.0j

    return chi


def evolucionar_un_paso(phi, alpha, N, s_tilde, V_tilde):
    """
    Evoluciona la función de onda un paso temporal usando el método de Cayley.

    La relación usada es:

        phi_{n+1} = chi_n - phi_n
    """

    beta = calcular_beta(phi, alpha, N, s_tilde, V_tilde)

    chi = calcular_chi(alpha, beta, N)

    phi_nuevo = chi - phi

    # Condiciones de contorno
    phi_nuevo[0] = 0.0 + 0.0j
    phi_nuevo[N] = 0.0 + 0.0j

    return phi_nuevo


# ============================================================
# 4. EVOLUCIÓN COMPLETA Y DETECTORES
# ============================================================

def prob_detector_izquierdo(phi, N):
    """
    Probabilidad de encontrar la partícula en el detector izquierdo.

    El detector izquierdo ocupa la región:
        j = 0, ..., N/5
    """

    fin = N // 5

    return np.sum(np.abs(phi[:fin + 1])**2)


def prob_detector_derecho(phi, N):
    """
    Probabilidad de encontrar la partícula en el detector derecho.

    El detector derecho ocupa la región:
        j = 4N/5, ..., N
    """

    inicio = 4 * N // 5

    return np.sum(np.abs(phi[inicio:])**2)


def evolucionar_simulacion(phi0, alpha, N, s_tilde, V_tilde, n_pasos, guardar_cada):
    """
    Evoluciona la función de onda durante n_pasos.

    Guarda:
        - densidades |phi|^2 cada guardar_cada pasos
        - norma en cada paso
        - probabilidad en detector izquierdo PI(t)
        - probabilidad en detector derecho PD(t)
    """

    phi = phi0.copy()

    tiempos_guardados = []
    densidades_guardadas = []

    tiempos = []
    normas = []
    PI = []
    PD = []

    phis_observables = []
    tiempos_observables = []

    for paso in range(n_pasos + 1):

        # Guardamos magnitudes en cada paso
        tiempos.append(paso)
        normas.append(norma(phi))
        PI.append(prob_detector_izquierdo(phi, N))
        PD.append(prob_detector_derecho(phi, N))

        # Guardamos la densidad solo cada cierto número de pasos
        if paso % guardar_cada == 0:
            tiempos_guardados.append(paso)
            densidades_guardadas.append(np.abs(phi)**2)

            tiempos_observables.append(paso)
            phis_observables.append(phi.copy())

        # Evolucionamos salvo en el último paso
        if paso < n_pasos:
            phi = evolucionar_un_paso(phi, alpha, N, s_tilde, V_tilde)

    return (
      np.array(tiempos),
      np.array(normas),
      np.array(PI),
      np.array(PD),
      np.array(tiempos_guardados),
      np.array(densidades_guardadas),
      np.array(tiempos_observables),
      np.array(phis_observables)
)


def guardar_densidades(nombre_archivo, j, V_tilde, tiempos_guardados, densidades_guardadas):
    """
    Guarda las densidades |phi|^2 en un archivo .dat.

    Formato:
        primera columna: j
        segunda columna: V_tilde
        siguientes columnas: |phi(j,t)|^2 para distintos tiempos
    """

    datos = np.column_stack((j, V_tilde, densidades_guardadas.T))

    nombres_columnas = "j V_tilde "
    nombres_columnas += " ".join([f"rho_t{t}" for t in tiempos_guardados])

    cabecera = (
        "Evolucion de la ecuacion de Schrodinger 1D\n"
        "Columnas:\n"
        f"{nombres_columnas}\n"
        f"N = {N}\n"
        f"n_ciclos = {n_ciclos}\n"
        f"lambda = {lam}\n"
        f"n_pasos = {n_pasos}\n"
        f"guardar_cada = {guardar_cada}\n"
    )

    np.savetxt(nombre_archivo, datos, header=cabecera)


def guardar_series_temporales(nombre_archivo, tiempos, serie, nombre_serie):
    """
    Guarda una serie temporal en un archivo .dat.

    Columnas:
        paso
        valor
    """

    datos = np.column_stack((tiempos, serie))

    cabecera = (
        f"Serie temporal: {nombre_serie}\n"
        "Columnas:\n"
        f"paso  {nombre_serie}\n"
    )

    np.savetxt(nombre_archivo, datos, header=cabecera)


def guardar_detectores(nombre_archivo, tiempos, PI, PD):
    """
    Guarda las probabilidades en los detectores izquierdo y derecho.
    """

    datos = np.column_stack((tiempos, PI, PD))

    cabecera = (
        "Probabilidades en los detectores\n"
        "Columnas:\n"
        "paso  PI  PD\n"
    )

    np.savetxt(nombre_archivo, datos, header=cabecera)


    # ============================================================
# 5. BÚSQUEDA DEL TIEMPO ÓPTIMO DE MEDIDA nD
# ============================================================

def encontrar_nD(tiempos, PD):
    """
    Busca el primer máximo local de PD(t).

    Un máximo local se detecta cuando:

        PD[i-1] < PD[i]  y  PD[i] > PD[i+1]

    Devuelve:
        nD: paso temporal correspondiente al primer máximo local
        PD_nD: valor de PD en ese paso
        indice_nD: índice del array donde ocurre el máximo
    """

    for i in range(1, len(PD) - 1):

        if PD[i - 1] < PD[i] and PD[i] > PD[i + 1]:

            nD = int(tiempos[i])
            PD_nD = PD[i]
            indice_nD = i

            return nD, PD_nD, indice_nD

    # Si no encuentra máximo local, usamos el máximo global como alternativa
    indice_nD = int(np.argmax(PD))
    nD = int(tiempos[indice_nD])
    PD_nD = PD[indice_nD]

    print("Aviso: no se encontró un máximo local claro.")
    print("Se usará el máximo global de PD como alternativa.")

    return nD, PD_nD, indice_nD


def guardar_nD(nombre_archivo, nD, PD_nD, indice_nD):
    """
    Guarda el tiempo óptimo de medida en un archivo .dat.
    """

    datos = np.array([[nD, PD_nD, indice_nD]])

    cabecera = (
        "Tiempo optimo de medida para el detector derecho\n"
        "Columnas:\n"
        "nD  PD(nD)  indice_nD\n"
    )

    np.savetxt(nombre_archivo, datos, header=cabecera)


    # ============================================================
# 6. MONTE CARLO PARA EL COEFICIENTE DE TRANSMISIÓN
# ============================================================

def estimar_K_montecarlo(PD_nD, m, seed=None):
    """
    Estima el coeficiente de transmisión K mediante Monte Carlo.

    Cada experimento se modela como una variable de Bernoulli:
        - éxito: la partícula se detecta a la derecha
        - probabilidad de éxito: PD(nD)

    Si r < PD(nD), contamos una transmisión.
    """

    rng = np.random.default_rng(seed)

    numeros_aleatorios = rng.random(m)

    transmisiones = numeros_aleatorios < PD_nD

    mT = np.sum(transmisiones)

    K = mT / m

    error_K = np.sqrt(K * (1.0 - K) / m)

    return K, error_K, int(mT)


def guardar_K(nombre_archivo, K, error_K, mT, m, PD_nD, nD):
    """
    Guarda el coeficiente de transmisión estimado por Monte Carlo.
    """

    datos = np.array([[K, error_K, mT, m, PD_nD, nD]])

    cabecera = (
        "Coeficiente de transmision estimado por Monte Carlo\n"
        "Columnas:\n"
        "K  error_K  mT  m  PD(nD)  nD\n"
    )

    np.savetxt(nombre_archivo, datos, header=cabecera)


    # ============================================================
# 7. VALORES ESPERADOS DE OBSERVABLES
# ============================================================

def valor_esperado_x(phi, N):
    """
    Calcula el valor esperado de la posición discreta j:

        <x> ≈ <j> = sum_j j |phi_j|^2

    Como la función de onda está normalizada, no dividimos por la norma.
    """

    j = np.arange(N + 1)

    rho = np.abs(phi)**2

    return np.sum(j * rho)


def valor_esperado_p(phi):
    """
    Calcula el valor esperado del momento discreto.

    Usamos el operador:

        p = -i d/dj

    y aproximamos la derivada con diferencias centrales:

        dphi/dj ≈ (phi_{j+1} - phi_{j-1}) / 2

    Solo se calcula en los puntos interiores.
    """

    dphi = np.zeros_like(phi, dtype=complex)

    dphi[1:-1] = (phi[2:] - phi[:-2]) / 2.0

    p_phi = -1j * dphi

    p_esp = np.sum(np.conjugate(phi) * p_phi)

    return np.real(p_esp)


def energia_cinetica(phi):
    """
    Calcula la energía cinética esperada discreta.

    En las unidades reescaladas del problema, el operador cinético es:

        T = - d²/dj²

    Usamos:

        d²phi/dj² ≈ phi_{j+1} - 2phi_j + phi_{j-1}

    y calculamos:

        <T> = sum_j phi_j^* (-d²phi/dj²)
    """

    d2phi = np.zeros_like(phi, dtype=complex)

    d2phi[1:-1] = phi[2:] - 2.0 * phi[1:-1] + phi[:-2]

    T_phi = -d2phi

    E_c = np.sum(np.conjugate(phi) * T_phi)

    return np.real(E_c)


def energia_potencial(phi, V_tilde):
    """
    Calcula la energía potencial esperada:

        <V> = sum_j V_j |phi_j|^2
    """

    rho = np.abs(phi)**2

    return np.sum(V_tilde * rho)


def energia_total(phi, V_tilde):
    """
    Calcula la energía total esperada:

        <E> = <T> + <V>
    """

    E_c = energia_cinetica(phi)

    E_v = energia_potencial(phi, V_tilde)

    return E_c + E_v


def calcular_observables(tiempos_observables, phis_observables, N, V_tilde):
    """
    Calcula los valores esperados de varios observables para los tiempos guardados.

    Devuelve arrays con:
        tiempo, <x>, <p>, <Ec>, <V>, <Etotal>
    """

    x_medios = []
    p_medios = []
    Ec_medias = []
    V_medias = []
    E_totales = []

    for phi in phis_observables:

        x_medios.append(valor_esperado_x(phi, N))
        p_medios.append(valor_esperado_p(phi))
        Ec_medias.append(energia_cinetica(phi))
        V_medias.append(energia_potencial(phi, V_tilde))
        E_totales.append(energia_total(phi, V_tilde))

    return (
        np.array(tiempos_observables),
        np.array(x_medios),
        np.array(p_medios),
        np.array(Ec_medias),
        np.array(V_medias),
        np.array(E_totales)
    )


def guardar_observables(nombre_archivo, tiempos_obs, x_medios, p_medios, Ec_medias, V_medias, E_totales):
    """
    Guarda los observables en un archivo .dat.

    Columnas:
        paso  <x>  <p>  <Ec>  <V>  <Etotal>
    """

    datos = np.column_stack((
        tiempos_obs,
        x_medios,
        p_medios,
        Ec_medias,
        V_medias,
        E_totales
    ))

    cabecera = (
        "Valores esperados de observables\n"
        "Columnas:\n"
        "paso  <x>  <p>  <Ec>  <V>  <Etotal>\n"
    )

    np.savetxt(nombre_archivo, datos, header=cabecera)


def guardar_estado_inicial(nombre_archivo, j, phi, V_tilde):
    """
    Guarda en un archivo .dat el estado inicial.

    Columnas:
        j
        Re(phi)
        Im(phi)
        |phi|^2
        V_tilde
    """

    datos = np.column_stack((
        j,
        np.real(phi),
        np.imag(phi),
        np.abs(phi)**2,
        V_tilde
    ))

    cabecera = (
        "Estado inicial para la ecuacion de Schrodinger 1D\n"
        "Columnas:\n"
        "j  Re(phi)  Im(phi)  |phi|^2  V_tilde\n"
        f"N = {N}\n"
        f"n_ciclos = {n_ciclos}\n"
        f"lambda = {lam}\n"
    )

    np.savetxt(nombre_archivo, datos, header=cabecera)

# ============================================================
# 5. PROGRAMA PRINCIPAL
# ============================================================

j, k_tilde, s_tilde = crear_parametros(N, n_ciclos)

V_tilde = crear_potencial_cuadrado(N, k_tilde, lam)

phi = crear_onda_inicial(N, k_tilde)

phi = normalizar(phi)

print("Parámetros de la simulación")
print("---------------------------")
print(f"N             = {N}")
print(f"n_ciclos      = {n_ciclos}")
print(f"lambda        = {lam}")
print(f"k_tilde       = {k_tilde:.6f}")
print(f"s_tilde       = {s_tilde:.6f}")
print(f"n_pasos       = {n_pasos}")
print(f"guardar_cada  = {guardar_cada}")
print()

print("Comprobación inicial")
print("--------------------")
print(f"Norma inicial = {norma(phi):.12f}")
print()

# Calculamos alpha una sola vez
alpha = calcular_alpha(N, s_tilde, V_tilde)

# Evolucionamos toda la simulación
tiempos, normas, PI, PD, tiempos_guardados, densidades_guardadas, tiempos_observables, phis_observables = evolucionar_simulacion(
    phi0=phi,
    alpha=alpha,
    N=N,
    s_tilde=s_tilde,
    V_tilde=V_tilde,
    n_pasos=n_pasos,
    guardar_cada=guardar_cada
)

print("Comprobación de conservación de la norma")
print("----------------------------------------")
print(f"Norma inicial       = {normas[0]:.12f}")
print(f"Norma final         = {normas[-1]:.12f}")
print(f"Desviación máxima   = {np.max(np.abs(normas - normas[0])):.3e}")
print()

print("Detectores")
print("----------")
print(f"Máximo PI = {np.max(PI):.6f}")
print(f"Máximo PD = {np.max(PD):.6f}")
print()


# Buscamos el primer máximo local de PD(t)
nD, PD_nD, indice_nD = encontrar_nD(tiempos, PD)

print("Tiempo óptimo de medida")
print("-----------------------")
print(f"nD          = {nD}")
print(f"PD(nD)      = {PD_nD:.6f}")
print(f"indice_nD   = {indice_nD}")
print()


# Estimamos el coeficiente de transmisión mediante Monte Carlo
K, error_K, mT = estimar_K_montecarlo(
    PD_nD=PD_nD,
    m=m,
    seed=seed
)

print("Coeficiente de transmisión")
print("--------------------------")
print(f"m                 = {m}")
print(f"mT                = {mT}")
print(f"K = mT/m          = {K:.6f}")
print(f"u(K) estadístico  = {error_K:.6f}")
print(f"PD(nD)            = {PD_nD:.6f}")
print(f"Diferencia K-PD   = {K - PD_nD:.6f}")
print()

# Calculamos valores esperados de observables
tiempos_obs, x_medios, p_medios, Ec_medias, V_medias, E_totales = calcular_observables(
    tiempos_observables=tiempos_observables,
    phis_observables=phis_observables,
    N=N,
    V_tilde=V_tilde
)

print("Observables")
print("-----------")
print(f"<x> inicial        = {x_medios[0]:.6f}")
print(f"<x> final          = {x_medios[-1]:.6f}")
print(f"<p> inicial        = {p_medios[0]:.6f}")
print(f"<p> final          = {p_medios[-1]:.6f}")
print(f"<Etotal> inicial   = {E_totales[0]:.6f}")
print(f"<Etotal> final     = {E_totales[-1]:.6f}")
print(f"Desviación Etotal  = {np.max(np.abs(E_totales - E_totales[0])):.3e}")
print()

# Guardamos archivos .dat
guardar_densidades(
    archivo_salida,
    j,
    V_tilde,
    tiempos_guardados,
    densidades_guardadas
)

guardar_series_temporales(
    archivo_normas,
    tiempos,
    normas,
    "norma"
)

guardar_detectores(
    archivo_detectores,
    tiempos,
    PI,
    PD
)

guardar_nD(
    archivo_nD,
    nD,
    PD_nD,
    indice_nD
)

guardar_K(
    archivo_K,
    K,
    error_K,
    mT,
    m,
    PD_nD,
    nD
)

guardar_observables(
    archivo_observables,
    tiempos_obs,
    x_medios,
    p_medios,
    Ec_medias,
    V_medias,
    E_totales
)


print("Archivos guardados")
print("------------------")
print(f"Densidades   -> {archivo_salida}")
print(f"Normas       -> {archivo_normas}")
print(f"Detectores   -> {archivo_detectores}")
print(f"nD           -> {archivo_nD}")
print(f"K            -> {archivo_K}")
print(f"Observables  -> {archivo_observables}")