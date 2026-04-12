Theory
======

Decoy-State BB84 Protocol
-------------------------

**Threshold detector model**:

.. math::

   Q_{\mu} &= Y_0 + 1 - e^{-\eta \mu}

   E_{\mu} Q_{\mu} &= e_0 Y_0 + e_d (1 - e^{-\eta \mu})

Two-Decoy Parameter Estimation
------------------------------

Lower bounds for security analysis (clipped [0,1]):

.. math::

   Y_0^L &= \frac{\nu_1 Q_{d2} e^{\nu_2} - \nu_2 Q_{d1} e^{\nu_1}}{\nu_1 - \nu_2}

   Y_1^L &= \frac{\mu}{(\nu_1-\nu_2)(\mu-(\nu_1+\nu_2))}
   \left[
   Q_{d1}e^{\nu_1} - Q_{d2}e^{\nu_2}
   - \frac{\nu_1^2 - \nu_2^2}{\mu^2}
   (Q_s e^{\mu} - Y_0^L)
   \right]

Single-photon phase error bound:

.. math::

   e_1^u = \frac{E_{d1}Q_{d1}e^{\nu_1} - E_{d2}Q_{d2}e^{\nu_2}}{(\nu_1-\nu_2) Y_1^L}

GLLP Secure Key Rate
--------------------

.. math::

   R &= \frac{1}{2} \left[ Q_1 (1 - h(e_1^u)) - Q_{\mu} f h(E_{\mu}) \right]

   Q_1 &= Y_1^L \mu e^{-\mu}