"""Webservice for single_wf_snip_analysis (Explore a single waveform with slang)"""

import base64

from flask import jsonify
from numpy import ndarray, array, frombuffer, ceil
import soundfile as sf

from peruse.single_wf_snip_analysis import TaggedWaveformAnalysis, DFLT_TILE_SIZE, DFLT_CHK_SIZE, DFLT_SR, LDA


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
        return self.get_attr_jdict(self.on_fit_return_attr)


########################################################################################################################
# py2http version

from py2http.service import run_http_service
from py2http.decorators import mk_flat, handle_json_req, handle_multipart_req

# Increase the request limit to be able to handle
from bottle import BaseRequest

max_n_minutes = 20
dflt_sr = 44100
n_bytes_per_sample = 2
BaseRequest.MEMFILE_MAX = 44100 * max_n_minutes * 60 * n_bytes_per_sample

input_trans_spec_00 = {
    'sr': int,
    'tile_size_frm': int,
    'chk_size_frm': int,
    'n_snips': int,
    'wf': array_or_str,
}


@handle_json_req  # extracts the JSON body and passes it to the input mapper as a dict
def array_input_mapper(input_kwargs):
    def gen():
        for k, v in input_kwargs.items():
            if k in input_trans_spec_00:
                yield k, input_trans_spec_00[k](v)
            else:
                yield k, v

    return dict(gen())


# from io import BytesIO
# import soundfile
#
# input_trans_spec = {
#     'sr': int,
#     'tile_size_frm': int,
#     'chk_size_frm': int,
#     'n_snips': int,
#     'wf': array_or_str,
# }
#
#
# def transform_with_mapping(input_kwargs, name_trans_mapping):
#     def gen():
#         for k, v in input_kwargs.items():
#             if k in name_trans_mapping:
#                 yield k, name_trans_mapping[k](v)
#             else:
#                 yield k, v
#
#     return dict(gen())
#
#
# @handle_multipart_req
# def sound_file_handling(input_kwargs):
#     input_kwargs = transform_with_mapping(input_kwargs, input_trans_spec)
#     file_input = input_kwargs.get('file_upload', None)
#     if file_input is not None:
#         filename = file_input.filename
#         file_bytes = BytesIO(file_input.file)
#         wf, sr = soundfile.read(file_bytes)
#         input_kwargs['wf'], input_kwargs['sr'] = wf, sr
#         return dict(input_kwargs, wf=wf, sr=sr, filename=filename)
#     else:
#         return input_kwargs


input_mapper = array_input_mapper
# input_mapper = sound_file_handling

fit = mk_flat(TaggedWaveformAnalysisForWS, TaggedWaveformAnalysisForWS.fit, func_name='fit')
fit.input_mapper = input_mapper
func_list = [fit]

from py2http import mk_http_service

app = mk_http_service(func_list)

########################################################################################################################
# Here starts where you need the http wrapper ##########################################################################

# from py2api.constants import _ARGNAME, _ELSE
# from py2api.py2rest.obj_wrap import WebObjWrapper
# from py2api.py2rest.input_trans import InputTrans
# from py2api.output_trans import OutputTrans
# from py2api.py2rest.app_maker import mk_app, dflt_run_app_kwargs
#
# # permissions ##########################################################################################################
#
# slang_inclusion_list = [
#     'fit',
# ]
#
# attr_permissions = {'include': slang_inclusion_list, 'exclude': []}
#
# input_trans = InputTrans(
#     trans_spec={
#         _ARGNAME: {
#             'sr': int,
#             'tile_size_frm': int,
#             'chk_size_frm': int,
#             'n_snips': int,
#             'wf': array_or_str
#         }
#     })
#
# output_trans = OutputTrans({_ELSE: lambda x: jsonify({'_result': x})})
#
# # wrapper ##############################################################################################################
# slang_lite = WebObjWrapper(obj_constructor=TaggedWaveformAnalysisForWS,
#                            obj_constructor_arg_names=['sr', 'tile_size_frm', 'chk_size_frm'],
#                            permissible_attr=attr_permissions,
#                            input_trans=input_trans,
#                            output_trans=output_trans,
#                            name='/slang_lite/',
#                            debug=0)
#
# # Adding routes to app #################################################################################################
#
# route_func_list = [slang_lite]
#
# module_name, _ = os.path.splitext(os.path.basename(__file__))
#
# app = mk_app(app_name=module_name, routes=route_func_list)

if __name__ == "__main__":
    # app_run_kwargs = dflt_run_app_kwargs()
    # app_run_kwargs['port'] = 5003
    # # print("Starting the app with kwargs: {}".format(app_run_kwargs))
    # app.run(**app_run_kwargs)

    # Create an HTTP server
    # run_http_service(func_list)
    app.run()
