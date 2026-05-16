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
   ChoiceTable.add_interaction :no-index:
   ChoiceTable.sample_alternatives :no-index:

Model Specification
-------------------

.. currentmodule:: locpick.spec

.. autosummary::
   :toctree: generated/

   ModelSpec :no-index:
   ParamRef :no-index:
   DataRef :no-index:
   InteractionTerm :no-index:
   ScopedTerm :no-index:
   P :no-index:
   X :no-index:
   interaction :no-index:

Models
------

.. currentmodule:: locpick.models.mnl

.. autosummary::
   :toctree: generated/

   MultinomialLogit :no-index:

.. currentmodule:: locpick.models.nested

.. autosummary::
   :toctree: generated/

   NestedLogit :no-index:
   NestSpec :no-index:
   NestingTree :no-index:

.. currentmodule:: locpick.models.mixed

.. autosummary::
   :toctree: generated/

   MixedLogit :no-index:
   ParamDistribution :no-index:

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
   WaldTest :no-index:
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
   MNLDataset :no-index: