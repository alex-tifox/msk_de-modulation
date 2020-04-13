# Python 3.8
# @author Tilnyi Oleksandr
# @subject KB
import numpy as np
import matplotlib.pyplot as plt

# GLOBAL VARS
# ------------
Tb = 0.1
fs = 1000
# ------------

# MODULATION
# -------------------------------
def msk_modulation(binary_string):
    print("Ciąg binarny %s" % binary_string)

    nrz_data = transform_to_nrz(binary_string)
    cophasal_bits_nrz, quadrature_bits_nrz = filter_input_nrz_data(nrz_data)

    cophasal_signal = generate_cophasal_signal(cophasal_bits_nrz)
    quadrature_signal = generate_quadrature_signal(quadrature_bits_nrz)
    modulated_signal = cophasal_signal + quadrature_signal
    draw_plot(modulated_signal, "Sygnał zmodulowany")

    return modulated_signal


def generate_cophasal_signal(cophasal_bits):
    cophasal_time = 2*Tb * len(cophasal_bits)-Tb

    rectifier_signal = np.abs(generate_wave_with_parameters(time_seconds=cophasal_time))
    draw_plot(rectifier_signal, "Prostownik")
    cophasal_bits = change_string_to_probes_in_time(cophasal_bits, Tb, cophasal_time)
    cophasal_signal = (np.array(cophasal_bits) * np.array(rectifier_signal))

    draw_plot(cophasal_signal, "Wymnożenie przez prostownik")
    cophasal_signal *= np.array(generate_wave_with_parameters(cophasal_time, f=10))

    draw_plot(cophasal_signal, "Synfazowy")
    return cophasal_signal


def generate_quadrature_signal(quadrature_bits):
    quadrature_time = 2*Tb * len(quadrature_bits)

    rectifier_signal = np.abs(generate_wave_with_parameters(time_seconds=quadrature_time))
    quadrature_bits = change_string_to_probes_in_time(quadrature_bits, Tb, quadrature_time)
    quadrature_signal = np.array(quadrature_bits) * np.array(rectifier_signal)
    quadrature_signal = np.insert(quadrature_signal, 0, np.zeros(int(Tb*fs)))

    draw_plot(quadrature_signal, "Wymnozenie przez prostownik")
    quadrature_signal *= generate_wave_with_parameters(quadrature_time+Tb, f=10, phi=np.pi/2)

    draw_plot(quadrature_signal, "Kwadraturowy")
    return quadrature_signal
# -------------------------------
# MODULATION END


# DEMODULATION
# -------------------------------
def msk_demodulation(msk_signal):
    msk_signal_time = len(msk_signal) / fs
    # print(msk_signal_time)
    # print(len(msk_signal))

    supporting_multiply_cophasal_result = np.array(msk_signal) * np.array(generate_wave_with_parameters(time_seconds=msk_signal_time, f=10))
    supporting_multiply_quadrature_result = np.array(msk_signal) * np.array(generate_wave_with_parameters(time_seconds=msk_signal_time, f=10, phi=np.pi/2))

    integrated_cophasal = integrate_cophasal_signal(supporting_multiply_cophasal_result)
    integrated_quadrature = integrate_quadrature_signal(supporting_multiply_quadrature_result)

    msk_demod_plot, axes = plt.subplots(32)
    axes[0][0].plot(msk_signal)

    axes[1][0].plot(integrated_cophasal)
    axes[1][1].plot(integrated_quadrature)
    # draw_plot(integrated_cophasal, "Synfazowy calkowany")
    # draw_plot(integrated_quadrature, "Kwadraturowy całkowany")

    filtered_cophasal, filtered_quadrature = filter_integration_find_cophasal_and_quadrature(integrated_cophasal, integrated_quadrature)

    axes[2][0].plot(filtered_cophasal)
    axes[2][1].plot(filtered_quadrature)
    # draw_plot(filtered_cophasal, "Filtered cophasal")
    # draw_plot(filtered_quadrature, "Filtered quadrature")
    plt.show()

    binary_cophasal = []
    binary_quadrature = []

    Tbp_for_loop = 2*Tb*fs
    for i in range(0, len(filtered_cophasal)):
        if i == Tbp_for_loop-1 or i+(Tb*fs) == Tbp_for_loop-1:
            if filtered_cophasal[i] == 1:
                binary_cophasal.append('1')
            elif filtered_cophasal[i] == 0:
                binary_cophasal.append('0')
            Tbp_for_loop += 2*Tb*fs

    Tbp_for_loop = 2*Tb*fs
    for i in range(0, len(filtered_quadrature)):
        if i == Tbp_for_loop-1:
            if filtered_quadrature[i] == 1:
                binary_quadrature.append('1')
            elif filtered_quadrature[i] == 0:
                binary_quadrature.append('0')
            Tbp_for_loop += 2 * Tb * fs

    # '11 00 01 11 11 0'
    # Cophasal - 1 0 0 1 1 0
    # Quadrature - 1 0 1 1 1
    # print(binary_cophasal)
    # print(binary_quadrature)

    binary_string_decoded = ''

    cophasal_iter = iter(binary_cophasal)
    quadrature_iter = iter(binary_quadrature)
    sum_length = len(filtered_cophasal) + len(filtered_quadrature)

    for i in range(0, sum_length-1):
        try:

            if i % 2 == 0:
                binary_string_decoded += str(next(cophasal_iter))
            elif i % 2 == 1:
                binary_string_decoded += str(next(quadrature_iter))
        except StopIteration:
            break

    # print(binary_cophasal)
    # print(binary_quadrature)
    print('Decoded binary is %s' % binary_string_decoded)


def integrate_cophasal_signal(signal):
    integrated_result = []
    Tb_probes = int(2*Tb*fs)
    for i in range(0, len(signal)+1):
        integrated_result.append(np.sum(signal[i-np.mod(i ,Tb_probes):i]))

    return integrated_result


def integrate_quadrature_signal(signal):
    integrated_result = []
    Tb_probes = int(2*Tb*fs)
    for i in range(0, len(signal) + 1):
        integrated_result.append(np.sum(signal[i - np.mod(i, Tb_probes):i]))

    return integrated_result


def filter_integration_find_cophasal_and_quadrature(cophasal_signal, quadrature_signal):
    filtered_cophasal = []
    filtered_quadrature = []
    for sig in cophasal_signal:
        if sig > 0:
            filtered_cophasal.append(1)
        elif sig <= 0:
            filtered_cophasal.append(0)

    for i in range(int(Tb*fs-1), len(quadrature_signal)):
        if quadrature_signal[i] > 0:
            filtered_quadrature.append(1)
        elif quadrature_signal[i] <= 0:
            filtered_quadrature.append(0)

    return filtered_cophasal, filtered_quadrature


# -------------------------------
# DEMODULATION END


def generate_wave_with_parameters(time_seconds=1, amplitude=1, fs=fs, f=2, phi=0):
    t = np.linspace(0, time_seconds, int(fs*time_seconds))
    return amplitude*np.sin(2*np.pi*t*f + phi)


def transform_to_nrz(binary_string):
    nrz_array = []
    for bin in binary_string:
        if bin == '1':
            nrz_array.append(1)
        elif bin == '0':
            nrz_array.append(-1)

    return nrz_array


def filter_input_nrz_data(nrz_data_string):
    cophasal_bits = []
    quadrature_bits = []

    for i in range(0, len(nrz_data_string)):
        # It seems that even bit is placed to cophasal array, not odd
        # But to the point of python-matlab view, even for us is at index 1-3-5,
        # odd - at indexes 0-2-4 etc
        if i % 2 == 0:
            cophasal_bits.append(nrz_data_string[i])
        else:
            quadrature_bits.append(nrz_data_string[i])

    return cophasal_bits, quadrature_bits


def change_string_to_probes_in_time(binary_string, Tb, time_seconds, fs=fs):
    t = np.linspace(0, time_seconds, int(fs*time_seconds))
    Tb = 2*Tb * fs
    Tb_for_loop = 0
    bin_str_iter = iter(binary_string)
    one_or_zero = int(next(bin_str_iter))
    result = []

    for i in range(0, len(t)):
        result.append(one_or_zero)
        if Tb_for_loop == i - Tb:
            if Tb_for_loop != len(t) - Tb+1:
                one_or_zero = int(next(bin_str_iter))
                Tb_for_loop += Tb

    draw_plot(result, "Ciąg NRZ: %s " % binary_string)
    return result


def draw_plot(data, title=""):
    plt.title(title)
    plt.plot(data)
    plt.show()


modulated_signal = msk_modulation('11000111110')
msk_demodulation(modulated_signal)
