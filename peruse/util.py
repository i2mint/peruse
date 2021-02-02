from functools import partial
import numpy as np
from numpy import ceil, zeros, hanning, fft
import matplotlib.pyplot as plt

DFLT_WIN_FUNC = hanning
DFLT_AMPLITUDE_FUNC = np.abs


def named_partial(name, func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    partial_func.name = name
    return partial_func


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


def chk_to_spectrum(chk, *, chk_size, window=DFLT_WIN_FUNC, amplitude_func=DFLT_AMPLITUDE_FUNC):
    assert len(chk) == chk_size, (
        f"This function was made for chk_size={chk_size}. "
        f"You fed a chk of size len(chk)={len(chk)} instead")
    fft_amplitudes = amplitude_func(np.fft.rfft(chk * window))
    return fft_amplitudes


def mk_chk_fft(chk_size=None, window=DFLT_WIN_FUNC, amplitude_func=DFLT_AMPLITUDE_FUNC):
    """Make a chk_fft function that will compute the fft (with given amplitude and window).
    Note that this output chk_fft function will enforce a fixed chk_size (given explicitly, or through the
    size of the window (if window is given as an array)

    >>> chk_size = 4 * 5
    >>> f = mk_chk_fft(chk_size)
    >>> chk = 4 * list(range(chk_size // 4))
    >>> f(chk)
    array([19.        , 10.214421  ,  0.40774689,  4.34150779,  8.09801978,
            4.30684381,  0.26711259,  2.61514496,  5.04588504,  2.6255033 ,
            0.21962565])
    >>> # verifying that it's pickable
    >>> import pickle
    >>> import numpy as np
    >>> ff = pickle.loads(pickle.dumps(f))
    >>> assert np.allclose(ff(chk), ff(chk))
    """
    if callable(window):
        assert chk_size is not None, "chk_size must be a positive integer if window is a callable, or None"
        window = window(chk_size)
    elif window is None:
        window = 1
    else:
        window = np.array(window)
        if chk_size is not None:
            assert len(window) == chk_size, f"chk_size ({chk_size}) and len(window) ({len(window)}) don't match"

    chk_spectrum = named_partial('chk_to_spectrum', chk_to_spectrum,
                                 chk_size=chk_size, window=window, amplitude_func=amplitude_func)
    chk_spectrum.chk_size = chk_size
    return chk_spectrum


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
