.. _api-covtype:

kernels.covtype package
=======================

.. automodule:: luas.kernels.covtype

The covariance objects returned by the kernel functions. Each class stores
one covariance matrix in a particular structure (dense, diagonal,
quasiseparable, ...) and implements the matrix operations that the 2D kernel
classes need, such as decompositions, matrix square roots and matrix-vector
products. You normally won't create these directly.

.. currentmodule:: luas.kernels.covtype

.. autosummary::
   :toctree: summary

   CovType
   General
   Identity
   ScaledIdentity
   Diagonal
   Exp
   GeneralQuasisep
   GeneralQuasisepPlusNoise
   PeriodicQuasisep
   Outer
   OuterPlusScaledIdentity
