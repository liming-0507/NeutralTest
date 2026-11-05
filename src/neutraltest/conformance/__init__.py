"""The bundled conformance suite.

These modules are the operator matrix shipped with the package: point pytest at
this directory (or run ``neutraltest test``) to check a backend against the CPU
``float64`` reference.  They are data, not framework code — adding coverage means
appending an ``OpCase``, never touching ``neutraltest`` itself.
"""
