.. _api_ref:

.. currentmodule:: locpick

API reference
=============

Data Pipeline
--------------

.. currentmodule:: locpick.data

.. autosummary::
   :toctree: generated/

   ChoiceArrays :no-index:
   ChoiceTable :no-index:
   EstimationProblem :no-index:

.. currentmodule:: locpick.data.choicetable

.. autosummary::
   :toctree: generated/

   ChoiceTable.from_tables :no-index:
   ChoiceTable.from_long :no-index:
   ChoiceTable.to_arrays :no-index:
   ChoiceTable.add_pairwise_variable :no-index:

Model Specification
-------------------

.. currentmodule:: locpick.spec

.. autosummary::
   :toctree: generated/

   ModelSpec :no-index:
   InteractionTerm :no-index:
   ScopedTerm :no-index:
   interaction :no-index:

Models
------

.. currentmodule:: locpick.models.choice_model

.. autosummary::
   :toctree: generated/

   ChoiceModel :no-index:

.. currentmodule:: locpick.models.nested

.. autosummary::
   :toctree: generated/

   NestSpec :no-index:
   NestingTree :no-index:

.. currentmodule:: locpick.models.mixed

.. autosummary::
   :toctree: generated/

   ParamDistribution :no-index:

.. currentmodule:: locpick.models.scl

.. autosummary::
   :toctree: generated/

   EdgeStructure :no-index:

Results
-------

.. currentmodule:: locpick.results.fit_result

.. autosummary::
   :toctree: generated/

   FitResult :no-index:

.. currentmodule:: locpick.results.diagnostics

.. autosummary::
   :toctree: generated/

   LikelihoodRatioTest :no-index:
   HausmanTest :no-index:
   WaldTest :no-index:
   lr_test :no-index:
   hausman_test :no-index:
   wald_test :no-index:

Sampling
--------

.. currentmodule:: locpick

.. autosummary::
   :toctree: generated/

   sample_alternatives :no-index:

Distance Utilities
-----------------

.. currentmodule:: locpick.data.distance

.. autosummary::
   :toctree: generated/

   distance_matrix :no-index:
   euclidean_distance_matrix :no-index:
   great_circle_distance_matrix :no-index:
   great_circle_vec :no-index:
   nearest_neighbors :no-index:
   distance_bands :no-index:
   pairwise_distance :no-index:

DGP Utilities
-------------

.. currentmodule:: locpick.dgp

.. autosummary::
   :toctree: generated/

   simulate_mnl :no-index:
   simulate_nested_logit :no-index:
   simulate_scl :no-index:
   simulate_mscl :no-index:
   simulate_nested_scl :no-index:
   simulate_mnscl :no-index:
   MNLDataset :no-index:
   NestedMNLDataset :no-index:
   SCLDataset :no-index:
   MSCLDataset :no-index:
   NestedSCLDataset :no-index:
   MNSCLDataset :no-index:

