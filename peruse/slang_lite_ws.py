"""Webservice for single_wf_snip_analysis (Explore a single waveform with slang)"""

import base64
import json

import numpy as np
from numpy import ndarray, array, frombuffer, ceil
from flask import jsonify

import soundfile as sf

from py2api.constants import _ARGNAME, _ELSE
from py2api.py2rest.obj_wrap import WebObjWrapper
from py2api.py2rest.input_trans import InputTrans
from py2api.output_trans import OutputTrans
from py2api.py2rest.app_maker import mk_app, dflt_run_app_kwargs

from peruse.single_wf_snip_analysis import TaggedWaveformAnalysis, DFLT_TILE_SIZE, DFLT_CHK_SIZE, DFLT_SR, LDA


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


to_jdict = NpEncoder().encode


def ensure_array(x):
    if not isinstance(x, ndarray):
        return array(x)
    else:
        return x


def array_or_str(x):
    if isinstance(x, (ndarray, str)):
        return x
    return array(x)


import os

MIN_BYTES_TO_BE_CONSIDERED_WAV_BYTES = 44 + 2 * int(DFLT_SR / DFLT_TILE_SIZE)  # header + one tile
B64_MIN_BYTES = int(4 * ceil(MIN_BYTES_TO_BE_CONSIDERED_WAV_BYTES / 3.0))


def type_of_wf(wf):
    if isinstance(wf, str):
        try:
            if wf.startswith('s3://'):
                return 'sref'
        except:
            pass

        try:
            if os.path.isfile(wf):
                return 'local_file'
        except:
            pass

        try:
            if len(wf) >= B64_MIN_BYTES:
                return 'wav_file_bytes'
        except:
            pass

        raise ValueError("Couldn't figure out the format of that (string) wf input.")
    elif isinstance(wf, (list, tuple, ndarray)):
        return "waveform"
    else:
        raise ValueError("Couldn't figure out the format of that wf input.")


def wf_from(wf):
    """
    Getting a waveform (int16 array) from various sources
    :param wf: a waveform, a bytesarray or string of int16 bytes, etc.
    :return: a waveform
    """
    wf_type = type_of_wf(wf)
    if wf_type == 'wav_file_bytes':
        raw = base64.b64decode(wf)
        return frombuffer(raw[44:], dtype='int16')
    elif wf_type == 'local_file':
        return sf.read(wf, dtype='int16')
    elif wf_type == 'sref':
        raise NotImplementedError("Not implemented sref wf_from yet.")
    else:
        return wf


DFLT_ON_FIT_RETURN_ATTR = ('snips', 'sr', 'chk_size_frm', 'tile_size_frm',
                           'n_snips', 'count_of_snip', 'tag_count_for_snip', 'knn_dict')


class TaggedWaveformAnalysisForWS(TaggedWaveformAnalysis):
    def __init__(self,
                 fv_tiles_model=LDA(n_components=11),
                 sr=DFLT_SR,
                 tile_size_frm=DFLT_TILE_SIZE,
                 chk_size_frm=DFLT_CHK_SIZE,
                 n_snips=None,
                 prior_count=1,
                 knn_dict_perc=15,
                 on_fit_return_attr=DFLT_ON_FIT_RETURN_ATTR
                 ):
        super(TaggedWaveformAnalysisForWS, self).__init__(
            fv_tiles_model=fv_tiles_model,
            sr=sr,
            tile_size_frm=tile_size_frm,
            chk_size_frm=chk_size_frm,
            n_snips=n_snips,
            prior_count=prior_count,
            knn_dict_perc=knn_dict_perc
        )
        self.on_fit_return_attr = on_fit_return_attr

    def fit(self, wf, annots_for_tag=None, n_snips=None):
        wf = wf_from(wf)
        super(TaggedWaveformAnalysisForWS, self).fit(wf, annots_for_tag=annots_for_tag, n_snips=n_snips)
        if self.snips is not None:  # TODO: Should be a web service concern!
            if isinstance(self.snips, ndarray):
                self.snips = self.snips.tolist()
        return to_jdict(self.get_attr_jdict(self.on_fit_return_attr))


# permissions ##########################################################################################################
slang_inclusion_list = [
    'fit',
]

attr_permissions = {'include': slang_inclusion_list, 'exclude': []}

input_trans = InputTrans(
    trans_spec={
        _ARGNAME: {
            'sr': int,
            'tile_size_frm': int,
            'chk_size_frm': int,
            'n_snips': int,
            'wf': array_or_str
        }
    })

output_trans = OutputTrans({_ELSE: lambda x: jsonify({'_result': x})})

# wrapper ##############################################################################################################
slang_lite = WebObjWrapper(obj_constructor=TaggedWaveformAnalysisForWS,
                           obj_constructor_arg_names=['sr', 'tile_size_frm', 'chk_size_frm'],
                           permissible_attr=attr_permissions,
                           input_trans=input_trans,
                           output_trans=output_trans,
                           name='/slang_lite/',
                           debug=0)

# Adding routes to app #################################################################################################

route_func_list = [slang_lite]

module_name, _ = os.path.splitext(os.path.basename(__file__))

app = mk_app(app_name=module_name, routes=route_func_list)

if __name__ == "__main__":
    app_run_kwargs = dflt_run_app_kwargs()
    app_run_kwargs['port'] = 5003
    # print("Starting the app with kwargs: {}".format(app_run_kwargs))
    app.run(**app_run_kwargs)
