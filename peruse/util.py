from numpy import ceil, zeros, hanning, fft

DFLT_WIN_FUNC = hanning


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
