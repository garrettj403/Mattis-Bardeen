Validation Data
===============

This directory has data pulled from different papers that can be used to validate ``mattisbardeen.py``.

This includes:

   - ``belitsky1995/``: V. Y. Belitsky, S. W. Jacobsson, L. V Filippenko, and E. L. Kollberg, “Theoretical and Experimental Studies of Nb-Based Tuning Circuits for THz SIS Mixers,” in Proceedings of the Sixth International Symposium on Space Terahertz Technology (ISSTT), 1995, pp. 87–102.
   
   - ``belitsky2006/``: V. Belitsky, C. Risacher, M. Pantaleev, and V. Vassilev, “Superconducting microstrip line model studies at millimetre and sub-millimetre waves,” Int. J. Infrared Millimeter Waves, vol. 27, no. 6, pp. 809–834, 2006.

   - ``reference/``: 40-digit values of the Mattis-Bardeen integrals computed with mpmath, independently of ``mattisbardeen.py`` (regenerate with ``python validation-data/reference/generate.py``, which needs ``mpmath``).

Other papers are checked against closed-form results directly in ``test_mattisbardeen.py`` (no data files): the T=0 conductivity of D. C. Mattis and J. Bardeen, Phys. Rev. 111, 412 (1958), Eqs. 3.11-3.13, and the finite-thickness surface impedances of A. R. Kerr, NRAO EDIR 302 (1996).
