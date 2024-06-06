import math
from collections import defaultdict, OrderedDict
from itertools import chain
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
__all__ = ['ProcessedSoundwave', 'FunctionSoundwave', 'Soundwave']


class BaseSoundwave:
    default_lowest = -50
    default_highest = 50

    def __init__(self, audio, t_counts=None):
        assert isinstance(audio, (np.ndarray, list, tuple)), "audio must be either numpy or python array"
        self.audio = audio
        if t_counts is None:
            t_counts = np.linspace(self.default_lowest, self.default_highest, len(self.audio))
        self.t_counts = t_counts

    @staticmethod
    def set_t_count_settings(lowest, highest):
        assert highest != lowest, "Specify the range where lowest < highest"
        highest = max(lowest, highest)
        lowest = min(lowest, highest)
        BaseSoundwave.default_highest = highest
        BaseSoundwave.default_lowest = lowest

    def extwrapper(self, t, diff_colors=False):
        assert len(self.audio) > 2, "Sample must contain more than 2 points"
        increasing = False if self.audio[1] < self.audio[0] else True
        average_min = 0
        average_max = 0
        max_total, min_total = 0, 0
        minima = []
        maxima = []

        for index in range(1, len(self.audio)):
            if self.audio[index] > self.audio[index - 1]:
                if increasing:
                    continue
                average_min += self.audio[index - 1]
                min_total += 1
                increasing = True
                minima.append((self.audio[index - 1], index - 1))

            elif self.audio[index] < self.audio[index - 1]:
                if not increasing:
                    continue
                average_max += self.audio[index - 1]
                max_total += 1
                increasing = False
                maxima.append((self.audio[index - 1], index - 1))
            else:
                continue

        assert isinstance(diff_colors, bool)
        if diff_colors:
            maxx = list(map(lambda x: x[0], maxima))
            maxy = list(map(lambda x: t[x[1]], maxima))
            minx = list(map(lambda x: x[0], minima))
            miny = list(map(lambda x: t[x[1]], minima))
            return [maxx, maxy], [minx, miny]

        sorted_extremums = sorted(chain(minima, maxima), key=lambda x: x[1])
        return list(map(lambda x: x[0], sorted_extremums)), list(map(lambda x: t[x[1]], sorted_extremums))

    def extaverage(self, *, audio=None):
        if audio is None:
            audio = self.audio
        assert len(audio) > 2, "Sample must contain more than 2 points"

        increasing = False if audio[1] < audio[0] else True
        average_min = 0
        average_max = 0
        max_total, min_total = 0, 0

        for index in range(1, len(audio)):
            if audio[index] > audio[index - 1]:
                if increasing:
                    continue
                average_min += audio[index - 1]
                min_total += 1
                increasing = True

            elif audio[index] < audio[index - 1]:
                if not increasing:
                    continue
                average_max += audio[index - 1]
                max_total += 1
                increasing = False
            else:
                continue
        if min_total:
            average_min /= min_total
        if max_total:
            average_max /= max_total
        return average_max, average_min

    def lstrip(self, threshold, partition_size=500):
        rem = len(self.audio) % partition_size

        clear_till = len(self.audio) - rem
        intervals = zip(reversed(np.arange(0, len(self.audio), partition_size)),
                        reversed(np.arange(partition_size, len(self.audio) - partition_size, partition_size)))
        if rem:
            rems = ((len(self.audio), len(self.audio) - rem) for _ in range(1))
            intervals = chain(rems, intervals)

        for rborder, lborder in intervals:
            avg_max, avg_min = self.extaverage(audio=self.audio[lborder:rborder])
            if avg_max > threshold and avg_min < -threshold:
                break
            if not avg_min or not avg_max:
                part_mean = np.mean(self.audio[lborder:rborder])
                if part_mean > threshold or part_mean < -threshold:
                    break
            clear_till -= partition_size
        return ProcessedSoundwave(self.audio[:clear_till])

    def rstrip(self, *, threshold, partition_size=500):
        rem = len(self.audio) % partition_size
        clear_till = 0
        remainder = ((len(self.audio) - rem, len(self.audio)) for _ in range(1))
        intervals = zip(range(0, len(self.audio), partition_size),
                        range(partition_size, len(self.audio) - partition_size, partition_size))

        for lborder, rborder in chain(intervals, remainder):
            avg_max, avg_min = self.extaverage(audio=self.audio[lborder:rborder])
            if avg_max > threshold or avg_min < -threshold:
                break
            if not avg_min or not avg_max:
                part_mean = np.mean(self.audio[lborder:rborder])
                if part_mean > threshold or part_mean < -threshold:
                    break
            clear_till += partition_size
        return ProcessedSoundwave(self.audio[clear_till:])


class ProcessedSoundwave(BaseSoundwave):

    def get_intervals(self, n):
        a = self._raw_intervals(n, self.audio)
     #   print(len(np.concatenate(a)))
        return a

    def _raw_intervals(self, n, sample):
        """Функиця, формирующая интервалы"""
        fn_max = max(sample) + 1 / 10 ** 5
        fn_min = min(sample)
        avg_max, avg_min = self.extaverage(audio=sample)
        avg_min_to_max = np.linspace(avg_min, avg_max, n, dtype=np.double)
        from_min_to_avg_min = np.linspace(fn_min, avg_min, n, dtype=np.double)
        from_avg_max_to_max = np.linspace(avg_max, fn_max, n, dtype=np.double)
        total = np.linspace(fn_min, fn_max, n, dtype=np.double)
        return from_min_to_avg_min, avg_min_to_max, from_avg_max_to_max, total

    def _prepare_intervals(self, n, sample=None):
        if sample is None:
            sample = self.audio
        from_min_to_avg_min, avg_min_to_max, from_avg_max_to_max, total = self._raw_intervals(n, sample)
        return total

    def binary_search_entropy(self, *, n):
        from collections import defaultdict
        intervals = self._prepare_intervals(n)
        intervals = list(map(
            lambda i: (intervals[i-1], intervals[i]), range(1, len(intervals))
        ))
        frequencies = defaultdict(int)
        for item in self.audio:
            low = 0
            high = len(intervals)
            while low <= high:
                mid = (low + high) // 2
                if intervals[mid][0] <= item < intervals[mid][1]:
                    frequencies[intervals[mid]] += 1
                    break
                else:
                    if intervals[mid][1] < item:
                        low = mid + 1
                    else:
                        high = mid - 1

        logger = logging.getLogger("BinarySearchEntropy")
        logger.info(f"""{len(frequencies)} intervals have signal values out of {len(intervals)}
        Total Probability={sum(np.fromiter(frequencies.values(), dtype=np.double) / len(self.audio))}""")
        psum = 0
        for p in frequencies.values():
            p /= len(self.audio)
            psum -= p * math.log2(p)

        logger.info(f"Total entropy: {psum}")
        return psum

    def _calculate_entropy(self, *, sample, n):
        intervals = self._prepare_intervals(sample=sample, n=n)
        audio = np.fromiter(sorted(sample), dtype=np.double)
        index = 0
        frequency = 0
        psum = 0
        interval_index = 1
        total_processed = 0

        freqs = defaultdict(float)
        probabilities = []
        ent = []

        while index < len(audio) and interval_index:
            if intervals[interval_index - 1] <= audio[index] < intervals[interval_index]:
                frequency += 1
            else:
                if frequency > 0:
                    p = frequency / len(audio)
                    probabilities.append(p)
                    total_processed += frequency
                    freqs[(intervals[interval_index - 1], intervals[interval_index])] = frequency
                    psum -= p * math.log2(p)
                    ent.append(p * math.log2(p))
                    frequency = 0
                else:
                    probabilities.append(0)
                    ent.append(0)
                interval_index += 1
                continue
            index += 1

        if frequency > 0:
            p = frequency / len(audio)
            probabilities.append(p)
            total_processed += frequency
            psum -= p * math.log2(p)
            ent.append(p * math.log2(p))
            freqs[(intervals[interval_index - 1], intervals[interval_index])] = frequency

        logger = logging.getLogger("StandardEntropyMethod")
        logger.info(f"""{len(freqs)} intervals have signal values out of {len(intervals) - 1}
        Total Probability={sum(np.fromiter(freqs.values(), dtype=np.double) / len(self.audio))}""")
        assert total_processed == len(audio), ("%d != %d" % (total_processed, len(audio)))
        assert sum(freqs.values()) >= 1 - 1 / 10**7
        logger.info(f"Total entropy: {psum}")
        return psum, freqs, probabilities, ent

    def total_entropy(self, *, n):
        res = self._calculate_entropy(sample=self.audio, n=n)
        return res


class Soundwave(BaseSoundwave):
    def filter(self, threshold, step):
        index_pairs = OrderedDict()
        recent_low = 0
        recent_high = 0
        for index in range(0, len(self.audio) - step - 1, step):
            avg_max, avg_min = self.extaverage(audio=self.audio[index: index + step])
            if avg_max < threshold and avg_min > -threshold:
                if index == recent_high:
                    recent_high = index + step
                else:
                    index_pairs[recent_low] = recent_high
                    recent_low = index
                    recent_high = index + step

        index_pairs[recent_low] = recent_high
        index_pairs[len(self.audio)] = 0
        ranges = list(index_pairs.items())
        result = []
        for i in range(1, len(ranges)):
            _, from_index = ranges[i - 1]
            to_index, _ = ranges[i]
            for j in range(from_index, to_index):
                result.append(self.audio[j])
        return ProcessedSoundwave(result)


class FunctionSoundwave(ProcessedSoundwave):

    def extwrapper(self, t=None, diff_colors=False):
        if t is None:
            return super().extwrapper(self.t_counts,  diff_colors=True)
        return super().extwrapper(t, diff_colors=True)
