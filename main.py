import numpy as np
from utils import audioread
import matplotlib.pyplot as plt
from entro import *
from display_utils import (
    AverageExtremumDisplayer,
    NDivisionDisplayer,
    EntropyCalculationDisplayer,
    Displayer
)


def main():
    t = np.linspace(-50, 50, 10000)
    t_squared = np.power(t, 2)
    y = np.sin(t) + np.abs(np.cos(t)) + np.abs(np.sin(t)) * np.power(2, np.cos(t_squared))
    #y = np.sin(2*np.pi*t)*np.sin(110*t)*np.cos(14*t)
    # y = np.cos(t)*np.abs(np.sin(t))
    n = 100

    fs3, audio3 = audioread('C://Users/11/Desktop/modelling/project/audio/sample.wav')
    Soundwave.set_t_count_settings(-100, 100)
    wave = Soundwave(audio3)
    wave = wave.filter(threshold=0.01, step=1000)
   # wave = FunctionSoundwave(y, t)

    Displayer(displayer=AverageExtremumDisplayer(wave))
    Displayer(displayer=NDivisionDisplayer(wave), n=n)
    Displayer(displayer=EntropyCalculationDisplayer(wave), n=n)
    plt.show()


if __name__ == '__main__':
   main()