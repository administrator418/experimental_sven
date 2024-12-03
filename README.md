# Beam Shaping using ML
This project aims at learning the correlation between a (manipulated / bad-shaped) beam and the phasemask that lead to this shape.

## Setup
- Install the required python libraries
    - tensorflow
    - matplotlib
    - numpy
- Ensure that `PATH_TO_STORE_RESULTS` points to an existing folder which shall be used for logging.
- Ensure that the variable `DATASET_PATH` points to your dataset folder which has the following structure:
    ```
    {DATASET_PATH}
    │
    │
    └───train
    │   │
    │   │
    │   └───beam_ff
    │   │      1.png
    │   │      2.png
    │   │      ...
    |   |
    │   └───beam_nf
    │   │      1.png
    │   │      2.png
    │   │      ...
    |   |
    │   └───phasemask
    │   │      1.png
    │   │      2.png
    │   │      ...
    │   
    └───test
    │   │
    │   │
    │   └───beam_ff
    │   │      1.png
    │   │      2.png
    │   │      ...
    |   |
    │   └───beam_nf
    │   │      1.png
    │   │      2.png
    │   │      ...
    |   |
    │   └───phasemask
    │   │      1.png
    │   │      2.png
    │   │      ...
    |
    └───val
        │
        │
        └───beam_ff
        │      1.png
        │      2.png
        │      ...
        |
        └───beam_nf
        │      1.png
        │      2.png
        │      ...
        |
        └───phasemask
               1.png
               2.png
               ...
    ```
    Note that the corresponding files need to have the same filename.

## Run
- Select your desired settings/hyperparameters at the top of `GAN.ipynb` for normal training in a Juypter Notebook or `GAN_hyperparameter_tuning.py` for hyperparameter tuning in a python environment.
- For hyperparameter tuning in a local environment, run `python GAN_hyperparameter_tuning.py` or `python3 GAN_hyperparameter_tuning.py`.
- For hyperparameter tuning in an AIME ML container, run `mlc-create gan-container Tensorflow 2.14.0 -w=/home/{USERNAME}/workspace/{PROJECT_PATH} -d=/home/data/{DATA_PATH}` to create and `mlc-open gan-container` to execute a container.
