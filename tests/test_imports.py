"""Smoke tests guarding ``peruse`` against silent upstream-API drift.

Two kinds of drift break this package without anything noticing:

* **Import-time drift** -- a module that ``peruse/__init__.py`` never reaches
  (``peruse.slang_lite_py2http_ws``, ``peruse.scrap.*``) still imports a name
  that an upstream dependency has since renamed. The module raises on import,
  and the wheel ships anyway.
* **Call-time drift** -- the module imports fine, and still builds its HTTP
  ``app``, but the calling convention it was written against has changed, so
  every request to that app fails at run time. Asserting the app merely *exists*
  does not catch this; only calling it does.

``test_module_imports`` covers the first, over a module list derived from the
package itself so that modules added later are covered automatically.
``test_fit_route_is_registered`` and ``test_fit_endpoint_answers_a_request``
cover the second, by driving the real WSGI app in-process -- no server, no port.

Note: importing ``peruse.slang_lite_py2http_ws`` sets
``bottle.BaseRequest.MEMFILE_MAX`` process-wide, a pre-existing module-scope side
effect that this suite inherits and never restores. A future test that depends on
bottle's request-size limit must snapshot and restore it itself.
"""

import importlib
import io
import json
import pkgutil

import numpy as np
import pytest

import peruse

# Derived, not hand-listed, so the docstring's claim stays true as modules are added.
# ``walk_packages`` yields a subpackage before importing it to recurse, so a subpackage
# that fails to import is still listed (and still fails below); only its children drop out.
MODULE_NAMES = ['peruse'] + sorted(
    module.name for module in pkgutil.walk_packages(peruse.__path__, 'peruse.')
)

PY2HTTP_MODULE_NAME = 'peruse.slang_lite_py2http_ws'

# ``TaggedWaveformAnalysis.fit`` falls back to a ``PCA(n_components=11)`` over the
# waveform's tiles when no annotations are given, so the test waveform needs at least
# that many tiles. A small margin keeps it clear of the floor and the test fast.
SR = 44100
TILE_SIZE_FRM = 1024
N_TILES = 12
N_SNIPS = 5


def _post_json(app, path, payload):
    """POST ``payload`` as JSON to ``path`` of a WSGI ``app``, in-process.

    Returns the ``(status, body)`` pair, ``body`` decoded from the JSON response.
    """
    encoded = json.dumps(payload).encode('utf-8')
    environ = {
        'REQUEST_METHOD': 'POST',
        'PATH_INFO': path,
        'SERVER_NAME': 'localhost',
        'SERVER_PORT': '80',
        'SERVER_PROTOCOL': 'HTTP/1.1',
        'CONTENT_TYPE': 'application/json',
        'CONTENT_LENGTH': str(len(encoded)),
        'wsgi.input': io.BytesIO(encoded),
        'wsgi.errors': io.BytesIO(),
        'wsgi.url_scheme': 'http',
    }
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured['status'] = status

    chunks = app(environ, start_response)
    return captured['status'], json.loads(b''.join(chunks).decode('utf-8'))


@pytest.mark.parametrize('module_name', MODULE_NAMES)
def test_module_imports(module_name):
    """Every module of the package imports without raising."""
    importlib.import_module(module_name)


def test_fit_route_is_registered():
    """The py2http app exposes the routes the service exists to serve."""
    app = importlib.import_module(PY2HTTP_MODULE_NAME).app
    routes = {(route.rule, route.method) for route in app.routes}
    assert {('/fit', 'POST'), ('/ping', 'GET')} <= routes


def test_fit_endpoint_answers_a_request():
    """POSTing a waveform to ``/fit`` returns snips rather than a 500.

    The route being mounted is not evidence that it works: the input mapper has
    its own calling convention with py2http, and a mismatch there fails only when
    a request actually arrives.
    """
    app = importlib.import_module(PY2HTTP_MODULE_NAME).app
    waveform = (np.random.RandomState(0).randn(N_TILES * TILE_SIZE_FRM) * 1000).astype(int)
    status, body = _post_json(
        app,
        '/fit',
        {
            'wf': waveform.tolist(),
            'sr': SR,
            'tile_size_frm': TILE_SIZE_FRM,
            'chk_size_frm': TILE_SIZE_FRM,
            'n_snips': N_SNIPS,
        },
    )
    assert status.startswith('200'), body
    assert len(body['snips']) == N_TILES
    assert set(body['snips']) <= set(range(N_SNIPS))
