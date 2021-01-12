"""Explore a single waveform with slang"""

import operator
import itertools
from collections import defaultdict, Counter, deque
from math import sqrt
from numpy import array, unique, log, ndarray, nan, where, empty, percentile, ravel

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans

from peruse.util import stft

MAX_N_SNIPS = 1001
DFLT_SR = 44100
DFLT_TILE_SIZE = 2048
DFLT_CHK_SIZE = DFLT_TILE_SIZE * 21


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


# NOTE: (more or less) copied from oto.trans.conversion to keep module independent
class ChkUnitConverter(object):
    def __init__(self, sr=DFLT_SR,
                 buf_size_frm=DFLT_TILE_SIZE,
                 chk_size_frm=DFLT_CHK_SIZE,
                 rounding_ndigits=6):
        self.sr = sr
        self.buf_size_frm = buf_size_frm
        self.chk_size_frm = chk_size_frm
        self.rounding_ndigits = rounding_ndigits

    def __call__(self, val, val_unit, target_unit):
        if target_unit == 'frm':
            if val_unit == 'buf':
                return int(val * self.buf_size_frm)
            elif val_unit == 's':
                return int(val * self.sr)
            elif val_unit == 'chk':
                return int(val * self.chk_size_frm)
        elif target_unit == 'buf':
            if val_unit == 'frm':
                return int(val / self.buf_size_frm)
            elif val_unit == 'chk':
                return round((val / self.buf_size_frm) * self.chk_size_frm, ndigits=self.rounding_ndigits)
        elif target_unit == 'chk':
            if val_unit == 's':
                return round(val * self.sr / self.chk_size_frm, ndigits=self.rounding_ndigits)
            elif val_unit == 'buf':
                return round((val * self.buf_size_frm) / self.chk_size_frm, ndigits=self.rounding_ndigits)
            elif val_unit == 'frm':
                return round(val / self.chk_size_frm, ndigits=self.rounding_ndigits)
        elif target_unit == 's':
            if val_unit == 'frm':
                return round(val / self.sr, ndigits=self.rounding_ndigits)
            elif val_unit == 'buf':
                return round(val * self.buf_size_frm / self.sr, ndigits=self.rounding_ndigits)
            elif val_unit == 'chk':
                return round(val * self.chk_size_frm / self.sr, ndigits=self.rounding_ndigits)
        raise ValueError("I didn't have a way to convert {} unit into {} unit.".format(val_unit, target_unit))


def abs_stft(wf, tile_size=DFLT_TILE_SIZE, tile_step=None):
    if len(wf) > 0:
        return abs(stft(wf, n_fft=tile_size, hop_length=tile_step))
    else:
        return array([])


def spectr(wf, sr=None, tile_size=DFLT_TILE_SIZE, tile_step=None):
    tile_step = tile_step or tile_size
    return abs_stft(wf, tile_size=tile_size, tile_step=tile_step).T


def log_spectr(wf, sr=None, tile_size=DFLT_TILE_SIZE, tile_step=None):
    return log(1 + spectr(wf, sr=sr, tile_size=tile_size, tile_step=tile_step))


def annots_type(annots):
    if isinstance(annots, dict):
        k = None
        v = None
        for k, v in annots.items():
            break
        if isinstance(k, str):
            if hasattr(v, '__len__') and len(v) > 0 and hasattr(v[0], '__len__') and len(v[0]) == 2:
                return 'tag_segments'
    else:
        item = annots[0]
        item_type = segment_tag_item_type(item)
        return item_type + '_list'
    raise ValueError("Can't recognize this type of annotation")


def segment_tag_item_type(item):
    if isinstance(item, dict) and 'tag' in item:
        if 'bt' in item and 'tt' in item:
            return 'bt_tt_tag_dict'
        elif 'segment' in item and len(item['segment']) == 2:
            return 'segment_tag_dict'
    elif len(item) == 3:
        if isinstance(item[0], str):
            # will not catch the case where tag is numerical (in that case we need to input the standard format!)
            return 'tag_bt_tt_tuple'
        else:
            return 'bt_tt_tag_tuple'  # the standard
    elif len(item) == 2:
        if hasattr(item[0], '__len__') and len(item[0]) == 2:
            return 'segment_tag_tuple'
        elif hasattr(item[1], '__len__') and len(item[1]) == 2:
            return 'tag_segment_tuple'
        else:
            raise TypeError("Can't recognize this time of annotation")
    else:
        raise TypeError("Can't recognize this type of annotation")


def tag_segments_dict_from_tag_segment_list(tag_segment_list):
    type_of_segment_tag = segment_tag_item_type(tag_segment_list[0])
    if type_of_segment_tag == 'segment_tag_tuple':
        it = tag_segment_list
    elif type_of_segment_tag == 'bt_tt_tag_dict':
        it = (((x['bt'], x['tt']), x['tag']) for x in tag_segment_list)
    elif type_of_segment_tag == 'segment_tag_dict':
        it = ((x['segment'], x['tag']) for x in tag_segment_list)
    elif type_of_segment_tag == 'bt_tt_tag_tuple':
        it = (((x[0], x[1]), x[2]) for x in tag_segment_list)
    elif type_of_segment_tag == 'tag_bt_tt_tuple':
        it = (((x[1], x[2]), x[0]) for x in tag_segment_list)
    elif type_of_segment_tag == 'tag_segment_tuple':
        it = ((x[1], x[0]) for x in tag_segment_list)
    else:
        raise TypeError("Can't recognize this type of tag_segment_list")

    tag_segments_dict = defaultdict(list)

    for (bt, tt), tag in it:
        tag_segments_dict[tag].append((bt, tt))

    return tag_segments_dict


def segment_tag_it_from_tag_segments_dict(tag_segments_dict):
    for tag, segment_list in tag_segments_dict.items():
        for bt, tt in segment_list:
            yield (bt, tt), tag


def count_to_prob(count_of_item, item_set, prior_count=1):
    prob_of_item = dict()
    total_count = 0
    for item in item_set:
        new_count = count_of_item.get(item, 0) + prior_count
        prob_of_item[item] = new_count
        total_count += new_count
    return {item: count / float(total_count) for item, count in prob_of_item.items()}


def key_aligned_val_op(dict_1, dict_2, op, dflt_val_1=None, dflt_val_2=None):
    result_dict = dict()
    for k, v1 in dict_1.items():
        v2 = dict_2.get(k, dflt_val_2)
        if v2 is not None:
            result_dict[k] = op(v1, v2)
            # else, don't include the key
    if dflt_val_1 is not None:
        for k in set(dict_2).difference(result_dict):
            result_dict[k] = op(dflt_val_1, dict_2[k])

    return result_dict


def key_aligned_val_op_with_forced_defaults(dict_1, dict_2, op, dflt_val_1, dflt_val_2):
    result_dict = dict()
    for k, v1 in dict_1.items():
        v2 = dict_2.get(k, dflt_val_2)
        result_dict[k] = op(v1, v2)
    for k in set(dict_2).difference(result_dict):
        result_dict[k] = op(dflt_val_1, dict_2[k])

    return result_dict


def dict_op_val(dict_1, val, op):
    return {k: op(v, val) for k, v in dict_1.items()}


class DictWithOps(dict):
    def __add__(self, dict_2):
        if isinstance(dict_2, dict):
            return key_aligned_val_op_with_forced_defaults(self, dict_2, operator.__add__, dflt_val_1=0, dflt_val_2=0)
        else:
            return dict_op_val(self, dict_2, operator.__add__)

    def __sub__(self, dict_2):
        if isinstance(dict_2, dict):
            return key_aligned_val_op_with_forced_defaults(self, dict_2, operator.__sub__, dflt_val_1=0, dflt_val_2=0)
        else:
            return dict_op_val(self, dict_2, operator.__sub__)

    def __mul__(self, dict_2):
        if isinstance(dict_2, dict):
            return key_aligned_val_op_with_forced_defaults(self, dict_2, operator.__mul__, dflt_val_1=1, dflt_val_2=1)
        else:
            return dict_op_val(self, dict_2, operator.__mul__)

    def __truediv__(self, dict_2):
        if isinstance(dict_2, dict):
            return key_aligned_val_op_with_forced_defaults(self, dict_2, operator.__truediv__, dflt_val_1=1,
                                                           dflt_val_2=1)
        else:
            return dict_op_val(self, dict_2, operator.__truediv__)


def normalized_cityblock_dist_mat(pts):
    n = len(pts)
    m = nan * empty((n, n))
    for i, v1 in enumerate(pts):
        for j, v2 in enumerate(pts):
            m[i, j] = sum(abs(v1 - v2))
    m /= m.max()
    return m


def knn_dict_from_dist_mat(dist_mat, p=15):
    row_idx, col_idx = where(dist_mat < percentile(dist_mat, p))
    mm = dist_mat[row_idx, col_idx]
    lidx = row_idx != col_idx
    row_idx = row_idx[lidx]
    col_idx = col_idx[lidx]
    mm = mm[lidx]
    knn = defaultdict(list)
    for i, _row_idx in enumerate(row_idx):
        knn[_row_idx].append((col_idx[i], mm[i]))
    return dict(knn)


def knn_dict_from_pts(pts, p=15):
    return knn_dict_from_dist_mat(normalized_cityblock_dist_mat(pts), p=p)


class TaggedWaveformAnalysis(object):
    def __init__(self,
                 fv_tiles_model=LDA(n_components=11),
                 sr=DFLT_SR,
                 tile_size_frm=DFLT_TILE_SIZE,
                 chk_size_frm=DFLT_CHK_SIZE,
                 n_snips=None,
                 prior_count=1,
                 knn_dict_perc=15
                 ):
        self.fv_tiles_model = fv_tiles_model
        self.sr = sr
        self.tile_size_frm = tile_size_frm
        self.chk_size_frm = chk_size_frm
        self.n_snips = n_snips

        self.fvs_to_snips = None
        self.tag_count_for_snip = None
        self.snips_of_fvs = None
        self.prior_count = prior_count
        self.tag_set = None
        self.convert_time = ChkUnitConverter(sr=self.sr,
                                             buf_size_frm=self.tile_size_frm,
                                             chk_size_frm=self.chk_size_frm)
        self.snips = None
        self.knn_dict_perc = knn_dict_perc
        self.knn_dict = None

    def fit(self, wf, annots_for_tag=None, n_snips=None):
        tiles, tags = self.log_spectr_tiles_and_tags_from_tag_segment_annots(wf, annots_for_tag)
        self.fit_fv_tiles_model(tiles, tags)
        fvs = self.fv_tiles_model.transform(tiles)
        self.fit_snip_model(fvs, tags, n_snips=n_snips)
        self.snips = self.snips_of_wf(wf)
        self.knn_dict = knn_dict_from_pts(self.fvs_to_snips.cluster_centers_, p=self.knn_dict_perc)
        return self

    def fit_from_tag_wf_iter(self, tag_wf_iter, n_snips=None):
        tiles, tags = self.log_spectr_tiles_and_tags_from_tag_wf_iter(tag_wf_iter)
        self.fit_fv_tiles_model(tiles, tags)
        fvs = self.fv_tiles_model.transform(tiles)
        self.fit_snip_model(fvs, tags, n_snips=n_snips)
        self.snips = ravel(list(map(self.snips_of_wf, tiles)))
        self.knn_dict = knn_dict_from_pts(self.fvs_to_snips.cluster_centers_, p=self.knn_dict_perc)
        return self

    def get_wf_for_bt_tt(self, wf, bt, tt):
        """
        Get waveform between bt and tt (both expressed in seconds)
        :param wf:
        :param bt:
        :param tt:
        :return:
        """
        # NOTE: DRY would dictate that we use ChkUnitConverter, but micro-service core not set up yet
        bf = int(round(bt * self.sr))
        tf = int(round(tt * self.sr))
        return wf[bf:tf]

    def tiles_of_wf(self, wf):
        return log_spectr(wf, tile_size=self.tile_size_frm)

    def log_spectr_tiles_and_y_from_wf(self, wf, tag):
        log_spectr_mat = self.tiles_of_wf(wf)
        return log_spectr_mat, [tag] * len(log_spectr_mat)

    def log_spectr_tiles_and_tags_from_tag_segment_annots(self,
                                                          wf,
                                                          tag_segments_dict=None
                                                          ):
        log_spectr_tiles = list()

        if tag_segments_dict is not None:
            tags = list()

            for tag, segments in tag_segments_dict.items():
                for bt, tt in segments:
                    tile_wf = self.get_wf_for_bt_tt(wf, bt, tt)
                    lst_tiles, tile_tags = self.log_spectr_tiles_and_y_from_wf(tile_wf, tag)
                    log_spectr_tiles += list(lst_tiles)
                    tags += list(tile_tags)

            return array(log_spectr_tiles), array(tags)
        else:
            return self.tiles_of_wf(wf), None

    def log_spectr_tiles_and_tags_from_tag_wf_iter(self, tag_wf_iter):
        log_spectr_tiles = list()

        tags = list()

        for tag, wf in tag_wf_iter:
            lst_tiles, tile_tags = self.log_spectr_tiles_and_y_from_wf(wf, tag)
            log_spectr_tiles += list(lst_tiles)
            tags += list(tile_tags)

        return array(log_spectr_tiles), array(tags)

    def fit_fv_tiles_model(self, tiles, tags=None):
        if tags is not None:
            self.fv_tiles_model.fit(tiles, tags)
        else:
            self.fv_tiles_model = PCA(n_components=11).fit(tiles)
        return self

    def fv_of_tiles(self, tiles):
        return self.fv_tiles_model.transform(tiles)

    def fit_snip_model(self, fvs, tags=None, n_snips=None):
        if n_snips is not None:
            self.n_snips = n_snips  # replace the object's n_snips parameter
        else:
            if self.n_snips is None:
                self.n_snips = int(min(MAX_N_SNIPS, max(2, round(sqrt(len(fvs))))))

        # fit a fv_to_snips model that will give us a snip
        self.fvs_to_snips = KMeans(n_clusters=self.n_snips).fit(fvs, y=tags)
        self.snips_of_fvs = self.fvs_to_snips.predict
        self.tag_count_for_snip = defaultdict(Counter)
        snips = self.snips_of_fvs(fvs)
        snips = snips.astype(int)

        if tags is not None:
            tag_list = getattr(self.fvs_to_snips, 'classes_', None)
            if tag_list:
                self.classes_ = tag_list
            else:
                self.classes_ = list(unique(tags))
            for snip, tag in zip(snips, tags):
                self.tag_count_for_snip[snip].update([tag])
            self.tag_count_for_snip = {snip: dict(counts) for snip, counts in self.tag_count_for_snip.items()}

        self.snips = None  # reinitialize (in case there was a fit before)

        return self

    # def snips_of_fvs(self, fvs):
    #     return self.fvs_to_snips.predict(fvs)

    def snip_of_fv(self, fv):
        return self.snips_of_fvs([fv])[0]

    def snips_of_wf(self, wf):
        tiles = self.tiles_of_wf(wf)
        fvs = self.fv_of_tiles(tiles)
        return self.snips_of_fvs(fvs)

    def snip_prob_from_count(self, count_of_snip):
        return count_to_prob(count_of_snip, item_set=list(self.count_of_snip.keys()), prior_count=self.prior_count)

    @lazyprop
    def count_of_snip(self):
        if self.snips is not None:
            return dict(Counter(self.snips))
        else:
            raise AttributeError("The object wasn't fit with sound yet (so I don't have snips to count)")

    @lazyprop
    def count_of_tag(self):
        c = Counter()
        for counts in list(self.tag_count_for_snip.values()):
            c.update(counts)
        return c

    # @lazyprop
    # def snip_prob(self):
    #     _snip_prob = pd.Series(self.count_of_snip)
    #     if self.n_snips is not None:  # then complete the _snip_prob with missing snips
    #         missing_snips = set(range(self.n_snips)).difference(_snip_prob.index.values)
    #         if missing_snips:
    #             _snip_prob = _snip_prob.append(pd.Series(data=1, index=missing_snips))
    #     _snip_prob /= sum(self.count_of_snip.values())
    #     return _snip_prob

    @lazyprop
    def prob_of_snip(self):
        total_count = sum(self.count_of_snip.values()) + len(self.count_of_snip) * self.prior_count
        return {snip: (count_of_snip + self.prior_count) / float(total_count)
                for snip, count_of_snip in self.count_of_snip.items()}

    @lazyprop
    def tag_prob_for_snip(self):
        return {snip: count_to_prob(tag_count, item_set=self.classes_, prior_count=self.prior_count)
                for snip, tag_count in self.tag_count_for_snip.items()}

    def tag_probs_for_snips(self, snips, tag=None):
        if tag is None:
            return array([self.tag_prob_for_snip[snip] for snip in snips])
        else:
            return array([self.tag_prob_for_snip[snip][tag] for snip in snips])

            # import pandas as pd
            # t = pd.DataFrame(self.tag_count_for_snip).fillna(self.prior_count)
            # return (t / t.sum(axis=0)).T

    def conditional_prob_ratio_of_snip(self, condition=None):
        if condition is None:
            # unif_prob_of_snip = 1 / self.n_snips
            condition = defaultdict(lambda: 0)  # uniform count of a snip
        else:
            if not isinstance(condition, dict):
                # then assume condition is a list of snips, and compute the snip count of it
                condition = Counter(condition)
                # else assume condition is already a snip count dict
        return DictWithOps(count_to_prob(condition, list(range(self.n_snips)))) / self.prob_of_snip

    def get_attr_jdict(self, attrs):
        if isinstance(attrs, str):
            attrs = [attrs]
        jdict = dict()
        for attr in attrs:
            val = getattr(self, attr, None)
            if isinstance(val, ndarray):
                val = list(val)
            elif isinstance(val, dict):
                val = list(val.items())
            jdict[attr] = val
        return jdict


def running_mean(it, chk_size=2, chk_step=1):  # TODO: A version of this with chk_step as well
    """
    Running mean (moving average) on iterator.
    Note: When input it is list-like, ut.stats.smooth.sliders version of running_mean is 4 times more efficient with
    big (but not too big, because happens in RAM) inputs.
    :param it: iterable
    :param chk_size: width of the window to take means from
    :return:

    >>> list(running_mean([1, 3, 5, 7, 9], 2))
    [2.0, 4.0, 6.0, 8.0]
    >>> list(running_mean([1, 3, 5, 7, 9], 2, chk_step=2))
    [2.0, 6.0]
    >>> list(running_mean([1, 3, 5, 7, 9], 2, chk_step=3))
    [2.0, 8.0]
    >>> list(running_mean([1, 3, 5, 7, 9], 3))
    [3.0, 5.0, 7.0]
    >>> list(running_mean([1, -1, 1, -1], 2))
    [0.0, 0.0, 0.0]
    >>> list(running_mean([-1, -2, -3, -4], 3))
    [-2.0, -3.0]
    """

    if chk_step > 1:
        # TODO: perhaps there's a more efficient way. A way that would sum the values of every step and add them in bulk
        yield from itertools.islice(running_mean(it, chk_size), None, None, chk_step)
    else:
        it = iter(it)
        if chk_size > 1:

            c = 0
            fifo = deque([], maxlen=chk_size)
            for i, x in enumerate(it, 1):
                fifo.append(x)
                c += x
                if i >= chk_size:
                    break

            yield c / chk_size

            if chk_step == 1:
                for x in it:
                    c += x - fifo[0]  # NOTE: seems faster than fifo.popleft
                    fifo.append(x)
                    yield c / chk_size
            else:
                raise RuntimeError("This should really never happen, by design.")
                # Below was an attempt at a faster solution than using the islice as is done above.
                # raise NotImplementedError("Not yet implemented (correctly)")
                # for chk in chunker(it, chk_size=chk_size, chk_step=chk_step, return_tail=False):
                #     print(chk)
                #     for x in chk:
                #         c += x - fifo.popleft()
                #     fifo.extend(chk)
                #     yield c / chk_size

        else:
            for x in it:
                yield x


from contextlib import suppress

with suppress(ModuleNotFoundError):
    import matplotlib.pylab as plt
    from hum import plot_wf  # pip install hum


    class TaggedWaveformAnalysisExtended(TaggedWaveformAnalysis):
        def plot_wf(self, x):
            plot_wf(x, self.sr)
            plt.grid('on')

        def plot_tiles(self, x, figsize=(16, 5), ax=None):
            plot_wf(x, self.sr / self.tile_size_frm, figsize=figsize, ax=ax)
            plt.grid('on')

        def plot_tag_probs_for_snips(self, snips, tag=None, smooth=None):
            t = self.tag_probs_for_snips(snips, tag)
            if smooth:
                t = list(running_mean(t, chk_size=smooth))
            self.plot_tiles(t)

########## For a webservice ############################################################################################
