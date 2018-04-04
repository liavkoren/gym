"""
From https://gist.github.com/sakulkar/f96a85b045638b399932eb1444d1baf9
"""
from itertools import product
from math import floor


class Quantizer(object):
    """Observation Qauntizer
    This class is used for quantizing the observations into discrete states to be used for QTable.QAgent
    """
    def __init__(self, low, high, buckets):
        """
        Parameters
        ----------
        low : List/tuple of Lowest possisble observation values
        high : List/tuple of Highest possisble observation values
        buckets : Number of buckets to quantize the dimension into (List or tuple)
        kwargs : Extra arguments passed (Not needed)
        -------
        """
        # static attributes
        self.low = low          # Lowest list of Observations
        self.high = high        # Highest list of Observations
        self.buckets = buckets  # Number of buckets per dimension
        self.dim = len(low)     # Dimension of the observation
        self.tuples = []
        # print(self.dim)

        self.width = []        # width of each quantization step
        for idx in range(self.dim):
            self.width.append((self.high[idx] - self.low[idx]) / self.buckets[idx])

    def quantize(self, state):
        """Quantize the state
        """
        quantized_obs = []
        for idx in range(self.dim):
            if state[idx] < self.low[idx]:
                quantized_obs.append(0)
            elif state[idx] >= self.high[idx]:
                quantized_obs.append(self.buckets[idx]-1)
            else:
                quantized_obs.append(int(floor((state[idx] - self.low[idx])/ self.width[idx])))
        return tuple(quantized_obs)

    def dequantize(self, quant_tuple):
        """ Provides a representative state for a given quant_tuple. """
        state = []
        for idx, dim_box in enumerate(quant_tuple):
            if dim_box == 0:
                state.append(self.low[idx] + self.width[idx]/2)
            elif dim_box == self.buckets[idx] - 1:
                state.append(self.high[idx] + self.width[idx]/2)
            else:
                state_val = dim_box * self.width[idx] + self.low[idx] + self.width[idx]/2
                state.append(state_val)
        assert self.quantize(state) == quant_tuple, f'computed state {state} does not match given tuple, expected {quant_tuple}, go {self.quantize(state)}'
        return tuple(state)


def tuples():
    out = []
    ranges = [list(range(bucket)) for bucket in [10, 10, 10, 10]]
    for tup in product(*ranges):
        out.append(tup)
    return out
