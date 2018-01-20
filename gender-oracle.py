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
    T = get_time_windows_amount(dataVoice, rate)
    dataVoice = align_data_voice(dataVoice, rate, T)
    partLen = int(rate)
    parts = [dataVoice[i*partLen:(i+1)*partLen] for i in range(int(T))]
    resultParts = [get_frequencies_intensities(data, rate) for data in parts if len(data) > 0]
    return oracle_gender(resultParts)


def get_frequencies_intensities(data, rate):
    window = np.hamming(len(data))
    data = data * window
    fftV = abs(fft(data)) / rate
    fftR = copy(fftV)
    for i in range(2, HPSLoop):
        tab = copy(fftV[::i])
        fftR = fftR[:len(tab)]
        fftR *= tab
    return get_only_valid_frequencies(fftR)


def align_data_voice(dataVoice, rate, T):
    return dataVoice[
            max(0, int(len(dataVoice) / 2) - int(T / 2 * rate)):
            min(len(dataVoice) - 1, int(len(dataVoice) / 2) + int(T / 2 * rate))
        ]


def get_time_windows_amount(dataVoice, rate):
    T = 3  # time for HPS method
    if T > len(dataVoice) / rate:
        T = len(dataVoice) / rate
    return T


def oracle_gender(resultParts):
    result = [0] * len(resultParts[0])
    for res in resultParts:
        result += res
    interval_width = 15
    interval_sums = [{'freq': i*interval_width, 'val':sum(result[interval_width*i:interval_width*i+14])}
                     for i in range(0, int(len(result)/interval_width))]
    most_intensive = oracle_by_most_intensive(interval_sums, interval_width)
    statistic_oracle = get_statistic_oracle(result)
    if statistic_oracle != most_intensive:
        aggregated = sum_with_intervals(interval_sums, interval_width)
        oracles = [most_intensive, statistic_oracle, aggregated]
        for_male = sum([1 for i in oracles if i == 1])
        for_female = sum([1 for i in oracles if i == 0])
        return for_male > for_female
    else:
        return statistic_oracle


def sum_with_intervals(interval_sums, interval_width):
    summed = [{'freq': i*interval_width,
               "val": interval_sums[i]['val'] + interval_sums[i+1]['val'] + interval_sums[i+2]['val']}
        for i in range(0, len(interval_sums) - 3)]
    sor = sort_by_val(summed)
    return oracle_by_freq(interval_width, sor)


def get_statistic_oracle(result):
    male = count_in_range(result, scaled_male)
    female = count_in_range(result, scaled_female)
    return int(male > female)


def oracle_by_most_intensive(intervals_sums, interval_width):
    sor = sort_by_val(intervals_sums)
    return oracle_by_freq(interval_width, sor)


def oracle_by_freq(interval_width, sor):
    freq = sor[0]['freq']
    if freq <= scaled_male[1]:
        return 1
    elif freq + interval_width - 1 >= scaled_female[0]:
        return 0
    else:
        return -1


def sort_by_val(intervals_sums):
    return sorted(intervals_sums, key=lambda i: i['val'], reverse=True)


def get_only_valid_frequencies(result):
    return result[humanVoiceMinMAx[0]:humanVoiceMinMAx[1]]


def count_in_range(arr, ranges):
    return sum(arr[ranges[0]:ranges[1]])


def show_fig(result):
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
        found= get_result(file)
        shouldBe = int(file.replace(".wav", "").endswith("M"))
        M[shouldBe][found] += 1
        if (shouldBe != found):
            print(file)

    print(M)
    wsp = (M[0][0] + M[1][1]) / (sum(M[0]) + sum(M[1]))
    print(wsp)
    # read_input()
