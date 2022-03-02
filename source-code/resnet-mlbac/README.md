## Updating the Current ML Model (_**F**<sub>current</sub>_) 

The `resnet-mlbac-finetuning.py` is the source code related to updating the Current ML Model (_**F**<sub>current</sub>_) 
and store the Updated ML Model (_**F**<sub>updated</sub>_) in the `results/` directory. 

The `resnet-mlbac-finetuning.py` takes three parameters in following order to incorporate of a Task (or a set of Tasks in case of multi-Tasks Administrataion).

- Current ML Model (_**F**<sub>current</sub>_) 
- AATs File Path
- ReplayData File Path

For example, `resnet-mlbac-finetuning.py` `../../trained_ml_models/initial_model-resnet-mlbac.hdf5` `../../datasets/single_task/aats_task1.sample` `../../datasets/single_task/replay_data_task1.sample`
is a sample command to **update** the current **initial_model-resnet-mlbac.hdf5** model using the AATs (`aats_task1.sample`) and ReplayData (`replay_data_task1.sample`) of Task **t-1**.

## Evaluating the Updated ML Model (_**F**<sub>updated</sub>_) 
The `resnet-mlbac-evaluation.py` is used to evaluate the performance of _**F**<sub>updated</sub>_ model. 
It takes two parameters in following order.

- Updated ML Model (_**F**<sub>updated</sub>_) 
- AATs (while evaluating for AATs) or OATs (while evaluating for OATs) File Path

For example, the `aats_task1.sample.hdf5` is the Updated ML Model (_**F**<sub>updated</sub>_) for Task **t-1**.
The following command will evaluate the updated model for OATs.

`resnet-mlbac-evaluation.py` `results/aats_task1.sample.hdf5` `../../datasets/single_task/oats_task1.sample`
