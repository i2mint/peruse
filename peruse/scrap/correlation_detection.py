from dataclasses import dataclass
from typing import Sequence, Optional, Iterable

from scipy.signal import correlate, find_peaks
from lined import Line, iterize
from slang import fixed_step_chunker
# from i2 import Sig

from peruse.util import named_partial

DFLT_CHK_SIZE = 2048


def mk_chunker(chk_size=DFLT_CHK_SIZE, chk_step=None, name='chunker'):
    if chk_step is None:
        chk_step = chk_size
    chunker = named_partial(name, fixed_step_chunker, chk_size=chk_size, chk_step=chk_step)
    chunker.chk_size = chk_size
    chunker.chk_step = chk_step
    return chunker


# wf_to_vols = Line(chunker, iterize(np.std))
# compute_template_corrs = Line(partial(correlate, template, mode='valid'),
#                               lambda x: x / len(template),
#                               )


class MotifDetector:
    """An object that detects locations of a sequence that match a motif sub-sequence."""

    def __init__(self, motif: Sequence, distance='motif_size', **find_peaks_kwargs):
        self.motif = motif
        self.motif_size = len(motif)
        if distance == 'motif_size':
            distance = self.motif_size
        self.distance = distance
        self.find_peaks_kwargs = dict(find_peaks_kwargs, distance=distance)

    def seq_to_match_scores(self, seq: Sequence):
        scores = correlate(self.motif, seq, mode='valid')
        return scores / self.motif_size

    def match_scores_to_peaks(self, scores: Sequence):
        peaks, peak_props = find_peaks(scores, **self.find_peaks_kwargs)
        return peaks

    def seq_to_peaks(self, seq: Sequence):
        return self.match_scores_to_peaks(self.seq_to_match_scores(seq))

    def __repr__(self):
        s = f"{type(self).__name__}(<motif of size {self.motif_size}>, "
        s += ', '.join(k + '=' + str(v) for k, v in self.find_peaks_kwargs.items()) + ')'
        return s

    __call__ = seq_to_peaks

