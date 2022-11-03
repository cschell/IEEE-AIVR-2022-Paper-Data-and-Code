import optuna
import pandas as pd
from optuna.trial import TrialState

STORAGE_URL = "postgresql://uaixdlamds_optuna:9lMadHgls2dt17M6E6xNs@132.187.8.222/uaixdlamds_optuna"
storage = optuna.storages.RDBStorage(url=STORAGE_URL)


study_summaries = storage.get_all_study_summaries()

all_studies = [(s.study_name, s._study_id) for s in study_summaries]


# study_name = "gru_brv_data"
# study_id = storage.get_study_id_from_name(study_name)

def trial_list_to_df(trial_list):
    return pd.DataFrame([t.__dict__ for t in trial_list])


for study_name, study_id in all_studies:
    print(f"processing trials from {study_name}")
    all_trials = trial_list_to_df(storage.get_all_trials(study_id=study_id))
    running_trials = trial_list_to_df(storage.get_all_trials(study_id=study_id, states=(TrialState.RUNNING,)))

    latest_trial_number = all_trials._number.max()

    if len(running_trials) == 0:
        continue

    stale_trials = running_trials[
        (running_trials._number != latest_trial_number) & \
        (running_trials._datetime_start < (pd.to_datetime("now") - pd.to_timedelta("2 days")))]

    for _, trial in stale_trials.iterrows():
        storage.set_trial_state(trial_id=trial._trial_id, state=TrialState.FAIL)
        print(f"    set trial {trial._number} to state 'fail'")
