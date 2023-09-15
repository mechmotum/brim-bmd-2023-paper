# BRiM: A Modular Bicycle-Rider Modeling Framework

## Generating Results
### Installation
The environment can be installed using `conda` or `mamba`:
```bash
conda env create -f brim-bmd-2023-paper.yml
```
The environment can be activated using:
```bash
conda activate brim-bmd-2023-paper
```
While a lock file will be added in the future, the commit hash of [`BRiM`] used in this
paper is `64e7d1832344cbcdac1e046725b7af411370da77`. Its associated `SymPy` commit hash
is `4e314f4c3c8d3cddaf7bcd30d9f2528b683ed317`.

### Running the Optimizations
A specific optimization can be run using:
```bash
python ./src/run_optimization.py -n <optimization_number>
```
where `<optimization_number>` is the number of the optimization to run.
The optimizations are numbered as follows:
- `1`: This is the default Whipple bicycle model with the `"Browser"` parametrization.
  The system's inputs are a torque applied to the rear wheel about the wheel hub axis
  and a torque actuator applied between the handlebars and the rear frame such that they
  are actuated about the steering axis. This optimization is referred to as the base
  case.
- `2`: This is the same as the base case but with the `"Pista"` parametrization.
- `3`: This is the same as the base case but with the `"Fisher"` parametrization.
- `4`: This modifies optimization `3` by replacing the rigid front frame with one with
  suspension.
- `5`: This is a bicycle-rider model. It extends the base case with a rigidly attached
  upper-body rider model, where the shoulders are modeled to allow flexion and rotation,
  and the elbows are pin joints.
- `6`: This modifies optimization `5` by replacing the steering torque with a pair of
  torque actuators at the elbows of the rider model.

To run all of the optimizations, use:
```bash
python ./src/run_optimization.py --all
```
 ### Creating the Figures
The figures can be created using:
```bash
python ./src/create_paper_figures.py
```

## File Structure
The file structure of the `src` folder is as follows:
- `data`: This directory contains the parametrization data from [`BicycleParameters`].
- `output`: This directory contains the output from the optimizations.
- `brim_extra.py`: This module contains additional shoulder connections, which are not
  yet in [`BRiM`].
- `container.py`: This module contains dataclasses for bookkeeping.
- `create_paper_figures.py`: This module creates the figures for the paper.
- `main.py`: This module contains the default parameters of the optimizations and can
  be easily modified and ran to solve custom optimizations.
- `model.py`: This module contains a function to create the bicycle-rider models based
  on the provided settings.
- `problem.py`: This module contains the functions to define the optimization problem
  and compute the initial guess.
- `run.py`: This module is a command line interface for running the optimizations,
  similar to `main.py`.
- `run_optimization.py`: This module is meant to be run from the command line and
  contains the logic for running the optimizations from the paper.
- `simulator.py`: This module contains a `Simulator` class, which is mostly copied
  from [`brim-examples`], to run the simulations.
- `utils.py`: This module contains several utility functions.

[`BicycleParameters`]: https://github.com/moorepants/BicycleParameters
[`BRiM`]: https://github.com/TJStienstra/brim
[`brim-examples`]: https://github.com/TJStienstra/brim-examples
