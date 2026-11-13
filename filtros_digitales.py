# ==========================================================
# Actividad: Diseño de Filtros Digitales
# Autor: [Cassandra Ramos Ramirez]
# Herramienta: Python + SciPy + Matplotlib
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---------- 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA ----------
fs = 1000  # Frecuencia de muestreo
t = np.arange(0, 1.0, 1/fs)

# Señal compuesta: 5 Hz + 50 Hz + 200 Hz + ruido blanco
np.random.seed(0)
x = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*50*t) \
    + 0.3*np.sin(2*np.pi*200*t) + 0.3*np.random.randn(len(t))

# ---------- 2. DISEÑO DEL FILTRO ----------
# Butterworth IIR orden 4
sos_low = signal.butter(4, 20, btype='lowpass', fs=fs, output='sos')
sos_high = signal.butter(4, 100, btype='highpass', fs=fs, output='sos')
sos_band = signal.butter(4, [30, 80], btype='bandpass', fs=fs, output='sos')

# FIR con ventana Hamming
fir = signal.firwin(101, 20, fs=fs, pass_zero='lowpass')

# Chebyshev Tipo I
sos_cheby = signal.cheby1(4, 1, 20, btype='lowpass', fs=fs, output='sos')

# ---------- 3. APLICACIÓN DEL FILTRO A LA SEÑAL ----------
y_low = signal.sosfilt(sos_low, x)
y_high = signal.sosfilt(sos_high, x)
y_band = signal.sosfilt(sos_band, x)
y_fir = signal.lfilter(fir, 1.0, x)
y_cheby = signal.sosfilt(sos_cheby, x)

# ---------- 4. VISUALIZACIÓN DE LOS RESULTADOS ----------
w, h_low = signal.sosfreqz(sos_low, worN=2000, fs=fs)
w, h_high = signal.sosfreqz(sos_high, worN=2000, fs=fs)
w, h_band = signal.sosfreqz(sos_band, worN=2000, fs=fs)

# Gráfica 1: Señales antes y después
fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
axs[0].plot(t, x); axs[0].set_title('Señal original: 5Hz + 50Hz + 200Hz + Ruido')
axs[0].grid(True)
axs[1].plot(t, y_low, color='green'); axs[1].set_title('Pasa Bajos 20 Hz - queda solo 5 Hz')
axs[1].grid(True)
axs[2].plot(t, y_high, color='red'); axs[2].set_title('Pasa Altos 100 Hz - queda solo 200 Hz')
axs[2].grid(True)
axs[3].plot(t, y_band, color='purple'); axs[3].set_title('Pasa Bandas 30-80 Hz - queda solo 50 Hz')
axs[3].grid(True)
axs[3].set_xlabel('Tiempo [s]')
plt.tight_layout(); plt.savefig('filtros.png'); plt.show()

# Gráfica 2: Respuesta en frecuencia
plt.figure(figsize=(10,4))
plt.plot(w, 20*np.log10(abs(h_low)), label='Pasa Bajos 20 Hz')
plt.plot(w, 20*np.log10(abs(h_high)), label='Pasa Altos 100 Hz')
plt.plot(w, 20*np.log10(abs(h_band)), label='Pasa Bandas 30-80 Hz')
plt.title('Respuesta en frecuencia'); plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]'); plt.grid(True); plt.legend()
plt.savefig('respuesta_frecuencia.png'); plt.show()

print("Listo: Filtros aplicados y gráficas generadas")