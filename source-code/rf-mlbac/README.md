## Updating the Current ML Model (_**F**<sub>current</sub>_) 

The `rf-mlbac-finetuning.py` is the source code related to updating the Current ML Model (_**F**<sub>current</sub>_) 
and store the Updated ML Model (_**F**<sub>updated</sub>_) in the `results/` directory. 

The `rf-mlbac-finetuning.py` takes three parameters in following order to incorporate of a Task (or a set of Tasks in case of multi-Tasks Administrataion).

- Current ML Model (_**F**<sub>current</sub>_) 
- AATs File Path
- ReplayData File Path

For example, python3 `rf-mlbac-finetuning.py` `../../trained_ml_models/initial_model-rf-mlbac.pkl` `../../datasets/single_task/aats_task1.sample` `../../datasets/single_task/replay_data_task1.sample`
is a sample command to **update** the current **initial_model-rf-mlbac.pkl** model using the AATs (`aats_task1.sample`) and ReplayData (`replay_data_task1.sample`) of Task **t-1**.


## Retraining a new ML Model (_**F**<sub>updated</sub>_) 

The `rf-mlbac-retraining.py` is the source code related to train a new model from scratch and and store the Updated ML Model (_**F**<sub>updated</sub>_) in the `results/` directory. 

The `rf-mlbac-retraining.py` takes two parameters in following order to incorporate of a Task (or a set of Tasks in case of multi-Tasks Administrataion).

- AATs File Path
- OATs File Path

For example, python3 `rf-mlbac-retraining.py` `../../datasets/single_task/aats_task1.sample` `../../datasets/single_task/replay_data_task1.sample`
is a sample command to **generate** a new model using the AATs (`aats_task1.sample`) and ReplayData (`replay_data_task1.sample`) of Task **t-1**.


## Evaluating the Updated ML Model (_**F**<sub>updated</sub>_) 
The `rf-mlbac-evaluate.py` is used to evaluate the performance of _**F**<sub>updated</sub>_ model. This program could evaluate both fine-tuning and retraining.
It takes two parameters in following order.

- Updated ML Model (_**F**<sub>updated</sub>_) 
- AATs (while evaluating for AATs) or OATs (while evaluating for OATs) File Path

For example, the `updated_rf_model.pkl` is the Updated ML Model (_**F**<sub>updated</sub>_) for Task **t-1**.
The following command will evaluate the updated model for OATs.

python3 `rf-mlbac-evaluate.py` `results/updated_rf_model.pkl` `../../datasets/single_task/oats_task1.sample`
