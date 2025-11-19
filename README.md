### INFORME DE LABORATORIO #5.
Variabilidad de la Frecuencia Cardíaca (HRV) y balance autonómico 
---------------
### OBJETIVOS
1. Investigar conceptos clave: sistema nervioso autónomo, HRV, ECG y diagrama de Poincaré.
2. Adquirir señal ECG en dos condiciones: reposo y lectura en voz alta.
3. Filtrar la señal ECG y extraer intervalos R-R.
4. Analizar HRV en el dominio del tiempo (media y desviación estándar).
5. Construir diagramas de Poincaré y calcular índices CVI y CSI
### PARTE A
# Sistema Nervioso Autónomo (SNA)

El **sistema nervioso autónomo** es la parte del sistema nervioso que inerva los órganos internos, incluidoslos vasos sanguíneos, el estómago, el intestino, el hígado, los riñones, la vejiga, los genitales, los pulmones, las pupilas, el corazón y las glándulas sudoríparas, salivales y digestivas.

Este sistema se divide en dos ramas principales:

- **Simpática**
- **Parasimpática**

## Sistema Nervioso Simpático

El sistema simpático prepara al organismo para situaciones de **estrés o emergencia**, conocidas como *lucha o huida*.

### Funciones principales
- Aumenta la frecuencia cardíaca.  
- Incrementa la fuerza de contracción del corazón.  
- Dilata las vías respiratorias.  
- Libera energía almacenada.  
- Aumenta la fuerza muscular.  
- Produce sudoración (especialmente en las palmas).  
- Dilata las pupilas.  
- Produce piloerección (erección del vello).
**Mecanismo:**
- Las fibras simpáticas liberan noradrenalina, que actúa sobre los receptores β₁-adrenérgicos del nodo sinoauricular (SA) del corazón.
- Esto incrementa la velocidad de despolarización espontánea de las células marcapasos, haciendo que los potenciales de acción se generen más rápidamente.

## Sistema Nervioso Parasimpático

El sistema parasimpático controla los procesos corporales durante situaciones ordinarias.  

### Funciones principales
- Se dedica a conservar y restaurar.  
- Retarda la frecuencia cardíaca.  
- Disminuye la presión arterial.
- Estimula el tubo digestivo para procesar los alimentos y eliminar los residuos.

**Mecanismo:**
- Las fibras parasimpáticas, principalmente del *nervio vago*, liberan **acetilcolina**, que actúa sobre los **receptores muscarínicos M₂** del nodo SA.
- Esto reduce la pendiente de despolarización espontánea y puede aumentar la **permeabilidad al potasio (K⁺)**, hiperpolarizando la membrana.

### SISTEMA NERVIOSO AUTÓNOMO  
![](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/Imagen1.jpg?raw=true)

# HRV
La variabilidad de la frecuencia cardíaca (HRV) se calcula a partir de la señal del electrocardiograma **(ECG)** midiendo las variaciones en el tiempo entre los latidos sucesivos, específicamente entre los **picos R**. Este análisis permite evaluar el funcionamiento del sistema nervioso autónomo y proporciona información sobre la capacidad de adaptación y recuperación del cuerpo. El proceso incluye identificar los picos R en la señal del ECG y luego calcular el intervalo entre ellos para obtener la señal de HRV. 

### Interpretación y significado de la HRV
- **HRV alta:** Generalmente se asocia con un sistema nervioso autónomo equilibrado, buena condición física y una mayor capacidad de adaptación al estrés.
- **HRV baja:** Puede indicar estrés, fatiga acumulada o un estado de sobreentrenamiento, ya que el cuerpo tiene una menor capacidad de adaptación.
- **Análisis de tendencias:** Es más valioso observar las tendencias de la HRV a lo largo del tiempo en lugar de un valor aislado, para así identificar patrones de recuperación y rendimiento. 

![](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/gr-hrv_Time1525275392438.jpg?raw=true)

# Diagrama de Poincaré 
El diagrama de Poincaré es una herramienta visual que permite analizar la variabilidad de los intervalos R–R del electrocardiograma, la cual refleja la interacción entre las ramas simpática y parasimpática del sistema nervioso autónomo que regulan el ritmo cardíaco.
En este gráfico, cada punto representa la relación entre dos intervalos R–R consecutivos. La forma y dispersión de los puntos aportan información sobre el equilibrio autonómico del corazón.

### Interpretación 
-	Una **alta variabilidad** (mayor dispersión en el diagrama de Poincaré) indica una predominancia parasimpática, asociada con un estado de reposo, buena adaptación fisiológica y equilibrio autonómico.
-	En cambio, una **baja variabilidad** (nube estrecha y alargada) sugiere una predominancia simpática, que refleja estrés, fatiga o menor capacidad de respuesta cardíaca.

![](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/Captura.JPG?raw=true)





### PARTE B
En esta etapa se aplica un filtro digital IIR para limpiar la señal ECG, se divide en dos segmentos de 2 minutos, y se detectan los picos R para calcular los intervalos R-R. Con esta información, se analizan parámetros básicos de la variabilidad de la frecuencia cardíaca (HRV) en el dominio del tiempo, como la media y la desviación estándar, comparando ambos segmentos para evaluar el balance autonómico.

### Desarrollo filtro IIR 
![](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/WhatsApp%20Image%202025-11-17%20at%201.04.20%20PM.jpeg?raw=true)
![](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/WhatsApp%20Image%202025-11-17%20at%201.04.52%20PM.jpeg?raw=true)
![](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/WhatsApp%20Image%202025-11-17%20at%201.07.15%20PM.jpeg?raw=true)
![MI IIR](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/mi%20IRR.png?raw=true)

La razon por la cual se presenta de esta manera el filtro IIR es debido a la inestabilidad numérica. Los coeficientes del denominador, obtenidos de la expansión algebraica de alto orden, tienen errores de redondeo. Este error provoca que al menos uno de los polos se ubique fuera del Círculo Unitario en el plano z, haciendo que el filtro sea inestable.

![IRR CON FUNCIONES PYTHON](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/IRR%20CON%20PYTHON.png?raw=true)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parámetros / archivo
# -----------------------------
archivo_csv = "senal_EcG_LAB52minsilencio.csv"   # cambia por tu archivo
columna_ecg = None               # None -> usa la última columna; o pon "ecg" o "signal" si viene por nombre

# -----------------------------
# Coeficientes (Fs = 500 Hz)
# -----------------------------
b0 = 0.20217668530533872
b1 = 0.0
b2 = -0.20217668530533872
a1 = -1.593074522250445
a2 = 0.5956466293893226

# -----------------------------
# Cargar CSV y seleccionar columna
# -----------------------------
df = pd.read_csv(archivo_csv)

if columna_ecg is None:
    # usar última columna por defecto
    ecg_col = df.columns[-1]
else:
    ecg_col = columna_ecg

x = df[ecg_col].astype(float).values

# -----------------------------
# Filtrado (ecuación en diferencias)
# y[n] = -a1*y[n-1] - a2*y[n-2] + b0*x[n] + b1*x[n-1] + b2*x[n-2]
# -----------------------------
y = np.zeros_like(x, dtype=float)
# estados iniciales son 0 (puedes cambiarlos si quieres)
x_1 = x_2 = 0.0
y_1 = y_2 = 0.0

for n, xn in enumerate(x):
    yn = -a1*y_1 - a2*y_2 + b0*xn + b1*x_1 + b2*x_2
    y[n] = yn
    # actualizar estados
    x_2 = x_1
    x_1 = xn
    y_2 = y_1
    y_1 = yn

# -----------------------------
# Guardar resultado
# -----------------------------
df_out = df.copy()
df_out[ecg_col + "_filtrado"] = y
out_name = "ecg_filtrado_fs500.csv"
df_out.to_csv(out_name, index=False)
print("Archivo guardado:", out_name)

# -----------------------------
# Graficar (opcional)
# -----------------------------
plt.figure(figsize=(10,4))
t = np.arange(len(x)) / 500.0  # tiempo en segundos (Fs=500)
plt.plot(t, x, label="ECG original", alpha=0.7)
plt.plot(t, y, label="ECG filtrado", alpha=0.8)
plt.legend()
plt.xlabel("Tiempo (s)")
plt.title("ECG: filtro pasabanda IIR (0.5-40 Hz), Fs=500 Hz")
plt.xlim(0, min(10, t[-1]))  # muestra primeros 10 s para mejor visualización
plt.show()


```

El gráfico inferior ("Filtro Butterworth de Python") muestra una implementación que utiliza coeficientes estables (generados por herramientas como "scipy.signal"). Ademas de esto se presenta Estabilidad Numérica . El código utiliza coeficientes calculados con alta precisión o, preferiblemente, implementa el filtro como una Cascada de Secciones de Segundo Orden (SOS), que es el método estándar para filtros IIR de alto orden. Esto asegura que todos los polos se mantengan dentro del Círculo Unitario, previniendo la inestabilidad.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal # Necesario para find_peaks

# =======================================================
# 1. PARÁMETROS DEL SISTEMA Y FILTRO (Mismos que los provistos)
# =======================================================
# Parámetros / archivo
archivo_csv = "senal_EcG_LAB52minsilencio.csv"  # ¡AJUSTA ESTE NOMBRE!
columna_ecg = None # None -> usa la última columna
Fs = 500  # Frecuencia de muestreo en Hz
T_SEGMENTO_MIN = 2  # Duración de cada segmento en minutos

# Coeficientes (Fs = 500 Hz) - Filtro IIR de 2do Orden estable
b0 = 0.20217668530533872
b1 = 0.0
b2 = -0.20217668530533872
a1 = -1.593074522250445
a2 = 0.5956466293893226


# =======================================================
# 2. CARGA Y FILTRADO DE LA SEÑAL (Implementación IIR $N=2$)
# =======================================================
try:
    df = pd.read_csv(archivo_csv, sep=',', decimal=',') # Ajuste para el formato de tu CSV
    
    if columna_ecg is None:
        ecg_col = df.columns[-1]
    else:
        ecg_col = columna_ecg

    x = df[ecg_col].astype(float).values
except Exception as e:
    print(f"Error al cargar el archivo: {e}. Asegúrate de que el nombre y la columna son correctos.")
    exit()

# Filtrado (ecuación en diferencias) - ASUME PARÁMETROS INICIALES EN 0
y_filtrado = np.zeros_like(x, dtype=float)
x_1 = x_2 = 0.0
y_1 = y_2 = 0.0

for n, xn in enumerate(x):
    # y[n] = -a1*y[n-1] - a2*y[n-2] + b0*x[n] + b1*x[n-1] + b2*x[n-2]
    yn = -a1 * y_1 - a2 * y_2 + b0 * xn + b1 * x_1 + b2 * x_2
    y_filtrado[n] = yn
    
    # actualizar estados (memoria)
    x_2 = x_1
    x_1 = xn
    y_2 = y_1
    y_1 = yn


# =======================================================
# 3. DIVISIÓN DE LA SEÑAL EN SEGMENTOS DE 2 MINUTOS
# =======================================================
N_segmento = T_SEGMENTO_MIN * 60 * Fs  # 2 min * 60 s/min * 500 Hz = 60000 muestras

# Asegurarse de que haya al menos dos segmentos
if len(y_filtrado) < N_segmento * 2:
    print("Advertencia: La señal es demasiado corta para dos segmentos de 2 minutos. Procesando lo que hay.")
    N_segmento = len(y_filtrado) // 2

segmento1 = y_filtrado[0:N_segmento]
segmento2 = y_filtrado[N_segmento:2*N_segmento]
print(f"Señal filtrada dividida en 2 segmentos de {N_segmento/Fs:.2f} segundos.")


# =======================================================
# 4. FUNCIÓN PARA IDENTIFICAR PICOS R y CALCULAR R-R
# =======================================================
def analizar_segmento(segmento, Fs, segmento_id):
    # Parámetros para find_peaks:
    # distance: Mínimo 0.3 segundos entre latidos (150 muestras a 500 Hz)
    # height: Umbral de 0.5 mV asumido para detectar el pico R, basado en visualizaciones anteriores.
    
    peaks, _ = signal.find_peaks(
        segmento, 
        height=0.5, 
        distance=int(0.3 * Fs)
    )
    
    if len(peaks) < 2:
        print(f"Segmento {segmento_id}: No se detectaron suficientes picos R para calcular R-R.")
        return []

    # Cálculo de Intervalos R-R (diferencia entre picos, en muestras)
    rr_muestras = np.diff(peaks)
    
    # Obtener una nueva señal con dicha información (RR en milisegundos)
    rr_ms = (rr_muestras / Fs) * 1000
    
    print(f"Segmento {segmento_id}: {len(peaks)} picos R detectados. {len(rr_ms)} intervalos R-R calculados.")
    
    return rr_ms


# =======================================================
# 5. EJECUCIÓN DEL ANÁLISIS Y CREACIÓN DE LA SEÑAL R-R
# =======================================================

rr_ms1 = analizar_segmento(segmento1, Fs, 1)
rr_ms2 = analizar_segmento(segmento2, Fs, 2)

# Unir los resultados en una sola señal
rr_ms_total = np.concatenate((rr_ms1, rr_ms2))
beat_number = np.arange(1, len(rr_ms_total) + 1)

# Crear la nueva señal R-R (DataFrame)
df_rr = pd.DataFrame({
    'Beat_Number': beat_number,
    'RR_Interval_ms': rr_ms_total
})

# Guardar y mostrar
out_rr_name = "RR_Interval_Signal_N2_Filtro.csv"
df_rr.to_csv(out_rr_name, index=False)
print(f"\nNueva señal de Intervalos R-R guardada en: {out_rr_name}")


# =======================================================
# 6. VISUALIZACIÓN DE LA NUEVA SEÑAL R-R
# =======================================================

plt.figure(figsize=(12, 6))
plt.plot(df_rr['Beat_Number'], df_rr['RR_Interval_ms'], marker='o', linestyle='-', color='darkgreen', markersize=3)
plt.title(f'Señal de Intervalos R-R (Total de {len(rr_ms_total)} latidos)')
plt.xlabel('Número de Latido (Beat Number)')
plt.ylabel('Intervalo R-R (ms)')
plt.grid(True)
plt.show()
```
![IRR ](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/imagen_2025-11-17_183340922.png?raw=true)

El proceso se implementó asumiendo que el filtro estaba en estado de reposo al inicio, con todos los parámetros iniciales en cero.

1. Filtrado Estable: Se aplicó el filtro IIR (N=2) con la ecuación en diferencias, asumiendo condiciones iniciales $0$
2. Segmentación: La señal filtrada se dividió en dos segmentos de 2 minutos cada uno.
3. Detección R-R: Se identificaron los picos R en cada segmento usando un umbral de altura y distancia.
4. Cálculo R-R: Se midió la diferencia de tiempo entre picos R consecutivos y se convirtió a milisegundos ($\text{ms}$), generando la señal final de Intervalos R-R.

#### Análisis del HRV

```python
import pandas as pd
import numpy as np

# --- PARAMETROS ---
FILENAME = "RR_Interval_Signal_N2_Filtro.csv"

# --- CÁLCULOS DE HRV EN EL DOMINIO DEL TIEMPO ---

def calcular_hrv_tiempo(rr_ms, nombre):
    """Calcula la media RR, SDNN y HR media para un segmento."""
    media_rr = np.mean(rr_ms)
    std_rr = np.std(rr_ms, ddof=1) # SDNN: Desviación Estándar de los Intervalos N-N
    media_hr = 60000 / media_rr # Frecuencia cardíaca media en BPM
    
    print(f"--- {nombre} ---")
    print(f"Latidos analizados: {len(rr_ms)}")
    print(f"Media RR (ms): {media_rr:.2f}")
    print(f"SDNN (ms): {std_rr:.2f}")
    print(f"Media HR (BPM): {media_hr:.2f}")
    return media_rr, std_rr, media_hr

# --- CARGAR Y DIVIDIR DATOS ---
try:
    df_rr = pd.read_csv(FILENAME)
    rr_ms_total = df_rr['RR_Interval_ms'].values
    N_total = len(rr_ms_total)
except Exception:
    print(f"Error al cargar {FILENAME}. Asegúrate de ejecutar el paso anterior primero.")
    exit()

# División: Asumimos una división simple por la mitad para fines de comparación,
# aunque la división exacta depende del número de latidos en los primeros 2 minutos.
N_rrs1 = N_total // 2 

rr_ms1 = rr_ms_total[:N_rrs1]
rr_ms2 = rr_ms_total[N_rrs1:]

# Ejecución
print("Realizando análisis de HRV...")
calcular_hrv_tiempo(rr_ms1, "Segmento 1")
calcular_hrv_tiempo(rr_ms2, "Segmento 2")
```
### Resultados
 ![HRV ](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/imagen_2025-11-17_184023148.png?raw=true)

 ### Análisis
#### Media R-R Y HR Media
Segmento 1 vs. Segmento 2: La frecuencia cardíaca promedio (HR) es ligeramente más alta en el Segmento 2 (103 BPM vs. 100 BPM). Esto sugiere que el sistema cardíaco estuvo levemente más activado o bajo menor influencia parasimpática durante el segundo periodo.
#### SDNN Y Balance autonómico
Segmento 1 (SDNN Bajo): La baja dispersión de los puntos en el primer segmento (Latidos = 1 a 200) indica un ritmo cardíaco relativamente estable y bajo control.
Segmento 2 (SDNN Alto): El valor de SDNN es drásticamente más alto. Esto se debe a los picos y valles extremos que aparecen después del latido 250.
Interpretación Biológica (Teórica): Un SDNN alto indica generalmente una gran flexibilidad autonómica y un buen control del Sistema Nervioso Autónomo (SNA).
Interpretación Práctica (Datos obtenidos): En tu caso, los picos son tan extremos (>1700 ms) que son casi seguro artefactos o errores de detección de picos R causados por el ruido persistente o por la suavidad del filtro IIR de segundo orden utilizado. Esto infla artificialmente el valor de SDNN, invalidando su uso como una medida biológica pura.








