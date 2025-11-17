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


### PARTE B
En esta etapa se aplica un filtro digital IIR para limpiar la señal ECG, se divide en dos segmentos de 2 minutos, y se detectan los picos R para calcular los intervalos R-R. Con esta información, se analizan parámetros básicos de la variabilidad de la frecuencia cardíaca (HRV) en el dominio del tiempo, como la media y la desviación estándar, comparando ambos segmentos para evaluar el balance autonómico.

### Desarrollo filtro IIR 
![](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/WhatsApp%20Image%202025-11-17%20at%201.04.20%20PM.jpeg?raw=true)
![](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/WhatsApp%20Image%202025-11-17%20at%201.04.52%20PM.jpeg?raw=true)
![](https://github.com/TomasCobos-rgb/INFORME-5-SENALES/blob/main/imagenes/WhatsApp%20Image%202025-11-17%20at%201.07.15%20PM.jpeg?raw=true)
![MI IIR]()

La razon por la cual se presenta de esta manera el filtro IIR es debido a la inestabilidad numérica. Los coeficientes del denominador, obtenidos de la expansión algebraica de alto orden, tienen errores de redondeo. Este error provoca que al menos uno de los polos se ubique fuera del Círculo Unitario en el plano z, haciendo que el filtro sea inestable.

![IRR CON FUNCIONES PYTHON]()

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
