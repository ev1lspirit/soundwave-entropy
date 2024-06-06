import abc
import matplotlib.pyplot as plt
import numpy as np
from entro import ProcessedSoundwave, FunctionSoundwave


class BaseDisplayer(abc.ABC):

    def __init__(self, wave: ProcessedSoundwave):
        self.wave = wave

    @abc.abstractmethod
    def display(self, *args, **kwargs):
        raise NotImplemented

    def draw_vertical_line(self, *, x, yrange):
        assert isinstance(yrange, tuple) and len(yrange) == 2
        borderx, bordery = np.zeros(sum(np.abs(yrange))), np.arange(min(yrange), max(yrange))
        borderx.fill(x)
        return borderx, bordery


def Displayer(displayer: BaseDisplayer, **kwargs):
    assert isinstance(displayer, BaseDisplayer), "Displayer must be a BaseDisplayer type"
    return displayer.display(**kwargs)


class AverageExtremumDisplayer(BaseDisplayer):

    def display(self, *args, **kwargs):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(self.wave.t_counts, self.wave.audio)
        y, t = self.wave.audio, self.wave.t_counts
        ax2.plot(t, y, color='grey', marker='*', ms=3)
        x0, y0 = self.wave.extwrapper(t, diff_colors=True)
        ax2.plot(y0[1], y0[0], color='blue', ms=15)
        ax2.plot(x0[1], x0[0], color='red', ms=15)
        return fig, ax1, ax2


class NDivisionDisplayer(BaseDisplayer):

    def display(self, *, n):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        avg_max, avg_min = self.wave.extaverage()
        intervals = self.wave.get_intervals(n)
        from_min_to_avg_min, avg_min_to_max, from_avg_max_to_max, total = intervals

        self.wave.binary_search_entropy(n=n)

        lborderx, lbordery = self.draw_vertical_line(x=avg_min, yrange=(-10, 11))
        ax1.plot(lborderx, lbordery)
        ax1.annotate('avg_min', (avg_min - 1 / 50, 10.3), fontsize=7)
        ax1.annotate('avg_max', (avg_max - 1 / 50, 10.3), fontsize=7)

        rborderx, rbordery = self.draw_vertical_line(x=avg_max, yrange=(-10, 11))
        ax1.plot(rborderx, rbordery, color='red')
        maxx, maxy = self.draw_vertical_line(x=max(self.wave.audio), yrange=(-10, 11))
        minx, miny = self.draw_vertical_line(x=min(self.wave.audio), yrange=(-10, 11))
        ax1.plot(maxx, maxy, color='green')
        ax1.annotate('max', (max(self.wave.audio) - 1 / 10 ** 2, 10.3), fontsize=7)
        ax1.plot(minx, miny, color='violet')
        ax1.annotate('min', (min(self.wave.audio) - 1 / 10 ** 2, 10.3), fontsize=7)

        ax1.scatter(avg_min_to_max, np.zeros(len(avg_min_to_max)), s=2, color='red')
        ax1.scatter(from_min_to_avg_min, np.zeros(len(from_min_to_avg_min)), s=5, color='green')
        ax1.scatter(from_avg_max_to_max, np.zeros(len(from_avg_max_to_max)), s=5, color='green')

        ax3.scatter(total,  np.zeros(len(total)), s=5, color='green')

        psum, freqs, probs, ent = self.wave.total_entropy(n=n)
        average = list(map(lambda u: np.mean(u), freqs.keys()))
        values = list(freqs.values())
        for val, text in zip(average, values):
            ax3.annotate(text, (val, 0), fontsize=6)

        ax2.plot(total, probs + [0])
        ax4.plot(total, ent + [0])
        return fig, ax1


class EntropyCalculationDisplayer(BaseDisplayer):

    def display(self, *, n):
        fig, (fax1, fax2) = plt.subplots(2, 1)
        fax1.plot(self.wave.t_counts, self.wave.audio, color="yellow")
        entropy = []
        splitters = []
        divisions = np.fromiter(
            map(round, np.linspace(30, len(self.wave.audio) - 1, 25)), dtype=np.int64)
        x_points = []

        for index, i in enumerate(divisions):
            ent, *_ = self.wave._calculate_entropy(n=n, sample=self.wave.audio[:i])
            entropy.append(ent)
            splitters.append(
                (np.full(25, self.wave.t_counts[i]),
                 np.linspace(min(self.wave.audio), max(self.wave.audio), 25)))
            x_points.append(i)

        for index, (x, y) in enumerate(splitters):
            fax1.plot(x, y, color="red")

        for index in range(len(splitters)-1):
            x, y = splitters[index]
            x1, y1 = splitters[index + 1]
            x_mean = np.mean([x, x1])
            y_mean = np.mean([y, y1])
            fax1.annotate(str(round(entropy[index], 2)), (x_mean-2, y_mean), fontsize=7, color="black")

        fax2.plot(x_points, entropy, marker='o', ms=4)
        return fig, fax1, fax2


