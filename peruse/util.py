from collections.abc import Callable
from functools import partial
import numpy as np
from numpy import ceil, zeros, hanning, fft
import matplotlib.pyplot as plt
from lined import Line

DFLT_WIN_FUNC = hanning
DFLT_AMPLITUDE_FUNC = np.abs
DFLT_WF_TO_SPECTRUM = Line(np.fft.rfft, abs)


def strongest_frequency_chk_size(wf, sr: int, wf_to_spectrum: Callable = DFLT_WF_TO_SPECTRUM):
    """Returns the chunk size that has the most energy in a fft transform.
    Meant to be used as an estimate of the pattern size in cases where size=step

    Usage note: The function performs a fft on the whole input waveform wf.
    Note: Function picks the frequency with the highest energy to determine the chunk size, but often there are
    several near-maximums around it. Perhaps we'd get a more accurate estimate by some statistics over
    several of the top frequencies (``np.argsort(-spectr)[:n]``)?

    """
    spectrum = wf_to_spectrum(wf)
    idx_of_strongest_frequency = np.argmax(spectrum)
    spectrum_chk_sizes = np.fft.rfftfreq(len(spectrum)) * sr
    return int(np.round(spectrum_chk_sizes[idx_of_strongest_frequency]))


def named_partial(name, func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    partial_func.name = name
    return partial_func


# TODO: Make equivalent using fft.rfft instead of fft.fft (which we don't need)
def stft(y, n_fft=2048, hop_length=None, win_func=DFLT_WIN_FUNC):
    '''
    :param y: Mx0 audio
    :param n_fft: window size
    :param hop_length: hop size
    :return: S - DxN stft matrix
    '''

    if hop_length is None:
        hop_length = n_fft

    if win_func is not None:
        win = win_func(n_fft)
    else:
        win = 1

    # calculate STFT
    M = len(y)
    N = int(ceil(1.0 * (M - n_fft) / hop_length) + 1)  # no. windows
    S = zeros((n_fft, N), dtype='complex')
    for f in range(N - 1):
        S[:, f] = y[f * hop_length:n_fft + f * hop_length] * win
    x_end = y[(N - 1) * hop_length:]
    S[:len(x_end), N - 1] = x_end
    S[:, N - 1] *= win
    S = fft.fft(S, axis=0)
    S = S[:n_fft // 2 + 1, :]

    return S


def pad_to_align(*arrays):
    """Yields arrays that are padded (on the right) with zeros so as to all be of the same length

    >>> x, y, z = pad_to_align([1, 2], [1, 2, 3, 4], [1, 2, 3])
    >>> x, y, z
    (array([1, 2, 0, 0]), array([1, 2, 3, 4]), array([1, 2, 3, 0]))

    """
    max_len = max(map(len, arrays))
    for arr in arrays:
        yield np.pad(arr, (0, max_len - len(arr)))


def pplot(*args, figsize=(22, 5), **kwargs):
    """The same old matplotlib plot, but with some custom defaults"""
    if figsize is not None:
        plt.figure(figsize=figsize)
    return plt.plot(*args, **kwargs)


def lazyprop(fn):
    """
    Instead of having to implement the "if hasattr blah blah" code for lazy loading, just write the function that
    returns the value and decorate it with lazyprop! See example below.

    Taken from https://github.com/sorin/lazyprop.

    :param fn: The @property method (function) to implement lazy loading on
    :return: a decorated lazy loading property

    >>> class Test(object):
    ...     @lazyprop
    ...     def a(self):
    ...         print 'generating "a"'
    ...         return range(5)
    >>> t = Test()
    >>> t.__dict__
    {}
    >>> t.a
    generating "a"
    [0, 1, 2, 3, 4]
    >>> t.__dict__
    {'_lazy_a': [0, 1, 2, 3, 4]}
    >>> t.a
    [0, 1, 2, 3, 4]
    >>> del t.a
    >>> t.a
    generating "a"
    [0, 1, 2, 3, 4]
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    @_lazyprop.deleter
    def _lazyprop(self):
        if hasattr(self, attr_name):
            delattr(self, attr_name)

    @_lazyprop.setter
    def _lazyprop(self, value):
        setattr(self, attr_name, value)

    return _lazyprop
