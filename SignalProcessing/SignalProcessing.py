import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft

script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

n = 500
Fs = 1000
F_max = 13

random_signal = np.random.normal(0, 10, n)
time_ticks = np.arange(n) / Fs

w = F_max / (Fs / 2)
filter_params = signal.butter(3, w, 'low', output='sos')
filtered_signal = signal.sosfiltfilt(filter_params, random_signal)

spectrum = fft.fft(filtered_signal)
spectrum_amplitude = np.abs(fft.fftshift(spectrum))
frequency_ticks = fft.fftfreq(n, 1 / Fs)
frequency_ticks_shifted = fft.fftshift(frequency_ticks)

def plot_and_save(x, y, xlabel, ylabel, title):
    fig_width = 21 / 2.54
    fig_height = 14 / 2.54
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    plt.tight_layout()
    file_name = f"{title}.png"
    file_path = os.path.join(figures_dir, file_name)
    fig.savefig(file_path, dpi=600)
    plt.close(fig)

signal_title = f"Сигнал з максимальною частотою F_max = {F_max} Гц"
plot_and_save(
    x=time_ticks, 
    y=filtered_signal, 
    xlabel="Час (секунди)", 
    ylabel="Амплітуда сигналу", 
    title=signal_title
)

spectrum_title = f"Спектр сигналу з максимальною частотою F_max = {F_max} Гц"
plot_and_save(
    x=frequency_ticks_shifted, 
    y=spectrum_amplitude, 
    xlabel="Частота (Гц)", 
    ylabel="Амплітуда спектру", 
    title=spectrum_title
)