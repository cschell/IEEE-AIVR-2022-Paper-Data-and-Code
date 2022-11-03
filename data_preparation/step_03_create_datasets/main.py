import yaml

from create_train_validation_datasets import create_train_validation_datasets
from determine_train_test_validation import determine_train_test_validation_takes

SETTINGS = yaml.safe_load(open("settings.yaml"))

take_assignments = determine_train_test_validation_takes(SETTINGS["database"])

output_path = "talking_with_hands_4_UAIXDLAMDS.hdf5"
print(f"selecting data from database and writing train, validation and test datasets to {output_path}")
create_train_validation_datasets(take_assignments, output_path, SETTINGS["database"])
print("done")