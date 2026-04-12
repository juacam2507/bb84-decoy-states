.. decoy-states-bb84 documentation master file, created by
   sphinx-quickstart on Sat Apr 11 13:07:30 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

decoy-states-bb84 QKD Simulator
===============================

**Fast, accurate decoy-state BB84 simulator** with threshold detectors, 
parameter estimation, and GLLP secure key rate computation.

Features
--------

- ✅ **Threshold detector model** with dark counts and misalignment
- ✅ **Two-decoy protocol** with Y₀ᴸ, Y₁ᴸ, e₁ᵘ bounds
- ✅ **GLLP secure key rate** computation
- ✅ **Monte Carlo simulation** vs theoretical validation
- ✅ **Full NumPy/SciPy implementation**

Quick Start
-----------

.. code-block:: bash

   uv init decoy-states-bb84
   cd decoy-states-bb84
   uv add numpy matplotlib scipy
   uv run python simulator.py # 10km, 10k iterations

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules

.. toctree::
   :maxdepth: 1
   :caption: Theory:

   theory

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
