## Syntax of the Dataset

*mlbac_access_control_state.sample* is the dataset of MLBAC administration. 

The dataset has around 5K users, 5K resources, eight user metadata, and eight resource metadata. Also, the dataset has four operations (op1, op2, op3, and op4).

As described in the paper, the dataset comprises a set of authorization tuples (samples).
The syntax of an authorization tuple in the dataset is:
unique id of a user u | unique id of a resource r | metadata values of all the user metadata of user u | metadata values of all the resource metadata of resource r | access information of all the four operations

A sample authorization tuple is:
`2396 2333 14 30 62 47 45 111 2 18 14 6 62 39 45 13 2 45 1 1 1 0`

This authorization tuple can be read as:
a user with uid `2396` has eight metadata, and their corresponding values are `14 30 62 47 45 111 2 18`.
A resource with rid `2333` has eight metadata, and their corresponding values are `14 6 62 39 45 13 2 45`.
The user has `op1`, `op2` and `op3` access to the resource as their corresponding binary digits are `1`. 
Also, the user does not have `op4` access on the resource as the respective binary flag is `0`.

## Tasks

We create eighteen disctinct Tasks based on the *mlbac_access_control_state* dataset for administration experiments. 

`single_task` directory contains the *AATs, OATs*, and *ReplayData* for each eighteen Tasks. We store them in seperate files. 
These are the datasets for the single-Task administration.

For multi-Task administration (two-Tasks, three-Tasks, and six-Tasks), we combine related *AATs, OATs*, and *ReplayData* for the sake of simplicity. 
The directories `two_tasks`, `three_tasks`, and `two_tasks` hold respective data for two-Tasks, three-Tasks, and six-Tasks, respectively.
