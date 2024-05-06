<img src="./docs/logos/EMIx.png" width="300" height="150">

Framework for parallel simulations of EMI and KNP-EMI models in complex geometries. 

```
@article{benedusi2024scalable,
  title={Scalable approximation and solvers for ionic electrodiffusion in cellular geometries},
  author={Benedusi, Pietro and Ellingsrud, Ada J and Herlyng, Halvor and Rognes, Marie E},
  journal={arXiv preprint arXiv:2403.04491},
  year={2024}
}
@article{benedusi2024modeling,
  title={Modeling excitable cells with the EMI equations: spectral analysis and iterative solution strategy},
  author={Benedusi, Pietro and Ferrari, Paola and Rognes, Marie E and Serra-Capizzano, Stefano},
  journal={Journal of Scientific Computing},
  volume={98},
  number={3},
  pages={58},
  year={2024},
  publisher={Springer}
}
```

## Dependencies

* FEniCS (www.fenicsproject.org)
* multiphenics (https://github.com/multiphenics/multiphenics)

## Quick start

The easiest way to get started is to install FEniCS via Docker and then to install multiphenics and the EMIx package within the Docker container by following these steps.

### Downloading the EMIx package

To download and enter the EMIx source code

```
git clone git@github.com:pietrobe/EMIx.git
cd EMIx
```

### Installing FEniCS via Docker

We recommend using Docker (www.docker.com) to install and run FEniCS. EMIx relies on FEniCS (legacy), and has been tested most recently with the specific Docker image listed below. To install FEniCS

```
sudo docker run -t -v $(pwd):/home/fenics -i ghcr.io/scientificcomputing/fenics:2023-11-15
cd /home/fenics
```

### Installing multiphenics

The multiphenics package adds robust parallel multi-domain support to (legacy) FEniCS. To install multiphenics

```
pip install git+https://github.com/multiphenics/multiphenics.git
```

### Installing EMIx package

To install the EMIx package

```
pip install -e .
```

### Testing the installation

To test that all dependencies are operating successfully using a
sample EMIx problem, run

```
cd examples
python3 EMI_example.py                 # Serial run
mpirun -n N python3 -u EMI_example.py  # Parallel run, with N the number of MPI processors used
```

The expected results are terminal output from 100 time steps of solving
the EMI equations and simulation results in a separate directory
(named output). For the KNP-EMI case, run:

```
mpirun -n N python3 -u KNPEMI_example.py
```

The input can be controlled via an *.yml* file. Further information about usage and input in *data/README.md* and *examples/data/README.md*.

## Contributing guidelines

**EMIx** is developed by [Pietro Benedusi](https://pietrobe.github.io/) in collaboration with [Marie E. Rognes](https://marierognes.org/)'s group at [Simula](https://www.simula.no/).

## Miscellaneous

If you run into memory issues in the form of segmentation faults or out-of-memory errors, check the memory limits of your Docker container, which may be set lower than the available system memory.

## Video
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ZMBpdS7VYNU/0.jpg)](https://www.youtube.com/watch?v=ZMBpdS7VYNU)

