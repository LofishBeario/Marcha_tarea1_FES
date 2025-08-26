import math
import numpy as np
import matplotlib.pyplot as plt


def marcha(n_pasos):
    """
    Genera una marcha aleatoria 1D.

    Parámetro:
    n_pasos (int)
        Número de pasos N.
        
    Retorna:
    np.ndarray
        Vector de posiciones x[0...N] con x_0=0.
    """
    step_length=1
    #Usamos el RNG que hay por defecto
    random = np.random.default_rng()
    pasos = random.choice([-step_length, step_length],size=n_pasos)
    pos = pasos.cumsum()
    return pos[-1]


def pos_finales(n_pasos, runs):
    """
    Repite la marcha aleatoria las veces que se indique y devuelve todas las posiciones finales.

    Parámetros:
    n_pasos : int
        Numero de pasos por trayectoria.
    runs : int
        Cantidad de corridas independientes.

    Retorna:
    np.ndarray
        Arreglo con las posiciones finales.
    """
    finales = np.empty(runs)
    for i in range(runs):
        x = marcha(n_pasos)
        finales[i] = x
    return finales


def histograma(posi_finales, n_pasos, bins=50):
    """
    Genera un histograma de las posiciones finales de las marchas aleatorias y lo compara con la distribución
    predicha por el teorema central del límite.

    La distribución segun el TCL para la posición final es:
        p_S ~ Normal(0, sigma^2),   con sigma = sqrt(N) * 1 (longitud del paso), y ⟨S_N⟩ = 0

    ParametrosL:
    posi_finales (array_like)
        Arreglo con las posiciones finales de múltiples corridas de la marcha aleatoria.
    n_pasos (int)
        Número de pasos (N) que se usó en cada corrida.

    Retorna
    Produce un gráfico.
    """
    step_length=1
    sigma = math.sqrt(n_pasos) * step_length

    plt.figure()
    #Histograma con density=True para que el area sea 1 y se pueda comparar con el TCL
    plt.hist(posi_finales, bins, density=True, alpha=0.6, label="Programa")

    #Curva gaussiana teórica
    x = np.linspace(min(posi_finales), max(posi_finales), 500)
    TCL = (1.0/(math.sqrt(2*math.pi)*sigma))*np.exp(-0.5*(x/sigma)**2)
    plt.plot(x, TCL, label="TCL")

    plt.title(f"Marcha aleatoria con N={n_pasos} y muestras={len(posi_finales)}")
    plt.xlabel("Posición final x")
    plt.ylabel("Densidad de probabilidad")
    plt.legend()
   
    plt.show()


def momentos_vs_N(valores_N, runs):
    """
    Calcula los momentos ⟨x⟩ y ⟨x^2⟩ de la marcha aleatoria para distintos valores de N, y estima 
    la constante de difusión D a partir de la relación:
        ⟨x^2⟩ = 2*D*N  (asumiendo Delta(t) = 1)

    Parametros:
    valores_N (lista de int)
        Lista de valores de N para los que se desea calcular
        ⟨x⟩ y ⟨x^2⟩.
    runs (int)
        Número de corridas independientes para cada N.

    Retorna:
    m1 (np.ndarray)
        Valores de ⟨x⟩ para cada N.
    m2 (np.ndarray)
        Valores de ⟨x^2⟩ para cada N.
    slope (float)
        Pendiente del ajuste lineal de ⟨x⟩ vs N. =a^2.
    D (float)
        Constante de difusión: D = slope/2.
    """
    medios=[]
    medios_x2=[]

    for N in valores_N:
        finales = pos_finales(N, runs)
        medios.append(finales.mean())
        medios_x2.append((finales ** 2).mean())
    N_arr = np.array(valores_N, dtype=float)
    m1 = np.array(medios)
    m2 = np.array(medios_x2)

    #Ajuste lineal m2 = slope*N + y_0
    slope, intercept = np.polyfit(N_arr, m2, 1)
    D = slope/2  # ¿porque <x^2> = 2*D*N (con Δt = 1)

    plt.figure()
    plt.scatter(N_arr,m2, label="Programa")
    plt.plot(N_arr, slope*N_arr+intercept, label=f"pendiente={slope}")
    plt.xlabel("N")
    plt.ylabel("⟨x^2⟩")
    plt.title(f"⟨x^2⟩ vs N")
    plt.legend()
    plt.show()

    return m1, m2, slope, D

def menu():
    print("Simulación de marcha aleatoria 1D")
    print("1) Una sola marcha")
    print("2) Histograma de posiciones finales vs TCL")
    print("3) Momentos ⟨x⟩ y ⟨x^2⟩ vs N")
    print("0) Salir")
    return input("Seleccione una opción: ")


def main():
    while True:
        opcion = menu()
        if opcion=="1":
            N=int(input("Número de pasos N: "))
            pos_final= marcha(N)
            print(f"Posición final: {pos_final}")

        elif opcion=="2":
            N=int(input("Número de pasos N: "))
            runs=int(input("Número de corridas: "))
            finales=pos_finales(N, runs)
            histograma(finales, N)

        elif opcion=="3":
            lista=input("Lista de valores de N separados por coma (100,300,600,1000): ")
            N_vals=[int(x) for x in lista.split(",")]
            runs=int(input("Número de corridas por cada N: "))
            medios,medios_x2,slope,D =momentos_vs_N(N_vals, runs)

            print("Resultados:")
            for N, u, v in zip(N_vals, medios, medios_x2):
                print(f"N={N:6d}  ⟨x⟩={u}  ⟨x^2⟩={v}")

            print(f"Pendiente del ajuste = {slope}")
            print(f"Constante de difusión estimada D = {D}")

        elif opcion=="0":
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()