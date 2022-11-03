from typing import List, Dict, Any

from optuna.distributions import BaseDistribution
from optuna.samplers import GridSampler
from optuna.study import Study
from optuna.trial import TrialState, FrozenTrial

from src.utils.utils import get_logger

_logger = get_logger(__name__)


class CustomGridSampler(GridSampler):
    def sample_relative(
            self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        # Instead of returning param values, GridSampler puts the target grid id as a system attr,
        # and the values are returned from `sample_independent`. This is because the distribution
        # object is hard to get at the beginning of trial, while we need the access to the object
        # to validate the sampled value.

        target_grids = self._get_unvisited_grid_ids(study)

        if len(target_grids) == 0:
            # This case may occur with distributed optimization or trial queue. If there is no
            # target grid, `GridSampler` exits the program.

            _logger.warning("search space is exhausted, exiting.")

            exit(0)

        return super().sample_relative(study, trial, search_space)

    def _get_unvisited_grid_ids(self, study: Study) -> List[int]:

        # List up unvisited grids based on already finished ones.
        visited_grids = []
        running_grids = []

        # We directly query the storage to get trials here instead of `study.get_trials`,
        # since some pruners such as `HyperbandPruner` use the study transformed
        # to filter trials. See https://github.com/optuna/optuna/issues/2327 for details.
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)

        for t in trials:
            if "grid_id" in t.system_attrs and self._same_search_space(
                    t.system_attrs["search_space"]
            ):
                if t.state in [TrialState.COMPLETE, TrialState.PRUNED]:
                    visited_grids.append(t.system_attrs["grid_id"])
                elif t.state == TrialState.RUNNING:
                    running_grids.append(t.system_attrs["grid_id"])

        unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids) - set(running_grids)
        # # If evaluations for all grids have been started, return grids that have not yet finished
        # # because all grids should be evaluated before stopping the optimization.
        # if len(unvisited_grids) == 0:
        #     unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids)

        return list(unvisited_grids)
