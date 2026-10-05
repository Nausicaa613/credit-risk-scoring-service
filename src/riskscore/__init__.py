"""riskscore - a dependency-free credit risk scoring service.

The package is organised in layers so that each concern can be tested in
isolation:

* ``config``    - runtime configuration resolved from the environment
* ``features``  - application payload -> numeric feature vector
* ``model``     - the linear logistic scorecard and its JSON artifact
* ``scorecard`` - model output -> score, risk band, decision, reason codes
* ``storage``   - SQLite persistence for applications
* ``audit``     - append-only audit trail
* ``metrics``   - in-process counters and latency histograms
* ``api``       - request routing, validation and JSON serialisation
* ``server``    - the HTTP transport built on the standard library

Nothing in this package imports a third-party module at runtime.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
