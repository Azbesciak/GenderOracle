import glob
import sys
from scipy import *
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

maleFemaleFreq = [120, 232]
TS = 3  # time for simple method

humanVoiceMinMAx = [60, 270]
maleMinMax = [60, 160]
femaleMinMax = [180, 270]
scaled_male = [v - humanVoiceMinMAx[0] for v in maleMinMax]
scaled_female = [v - humanVoiceMinMAx[0] for v in femaleMinMax]
HPSLoop = 5
figure_show = False


def HPS(rate, dataVoice):
    T = 3  # time for HPS method

    if T > len(dataVoice) / rate:
        T = len(dataVoice) / rate
    dataVoice = dataVoice[
                max(0, int(len(dataVoice) / 2) - int(T / 2 * rate)):
                min(len(dataVoice) - 1, int(len(dataVoice) / 2) + int(T / 2 * rate))
                ]
    partLen = int(rate)
    parts = [dataVoice[i * partLen:(i + 1) * partLen] for i in range(int(T))]
    resultParts = []
    for data in parts:
        if len(data) != 0:
            window = np.hamming(len(data))
            data = data * window
            fftV = abs(fft(data)) / rate
            fftR = copy(fftV)
            for i in range(2, HPSLoop):
                tab = copy(fftV[::i])
                fftR = fftR[:len(tab)]
                fftR *= tab
            resultParts.append(fftR)
    return oracle_gender(resultParts)


def oracle_gender(resultParts):
    result = [0] * len(resultParts[int(len(resultParts) / 2)])
    for res in resultParts:
        if len(res) == len(result):
            result += res
    result = result[humanVoiceMinMAx[0]:humanVoiceMinMAx[1]]
    delta = max(result) - min(result)
    # result[result < (delta * 0.2) + min(result)] = 0
    show_fig(result)
    male = count_in_range(result, scaled_male)
    female = count_in_range(result, scaled_female)
    # print(male, female, male - female)
    return int(male > female), male, female, abs(male-female)


def count_in_range(arr, ranges):
    return sum(arr[ranges[0]:ranges[1]])

def show_fig(result):
    if figure_show:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(result)
        fig.show()


def get_result(file):
    sound_file, freq_width = sf.read(file)
    first_chanel_only = get_first_chanel(sound_file)
    return HPS(freq_width, first_chanel_only)


def get_first_chanel(tab):
    if type(tab[0]) in (tuple, list, np.ndarray):
        first_chanel = [x[0] for x in tab[:]]
        return first_chanel
    else:
        return tab


def read_input():
    # if len(sys.argv) < 2:
    #     print("missing path to file", file=sys.stderr)
    #     return
    # file_name = sys.argv[1]
    file_name = "train/001_K.wav"
    result = get_result(file_name)
    print("K" if result == 0 else "M")


if __name__ == "__main__":
    # male: 1 female: 0
    M = [[0, 0], [0, 0]]
    files = glob.glob("train/*.wav")
    for file in files:
        found, male, female, dif = get_result(file)
        shouldBe = int(file.replace(".wav", "").endswith("M"))
        M[shouldBe][found] += 1
        if (shouldBe != found):
            print(file, male, female, dif)

    print(M)
    wsp = (M[0][0] + M[1][1]) / (sum(M[0]) + sum(M[1]))
    print(wsp)
    # read_input()
