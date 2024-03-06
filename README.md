<img src="./docs/logos/EMIx.png" width="300" height="150">

Framework for parallel simulations of EMI and KNP-EMI models in complex geometries. 

## Dependencies

* FEniCS (www.fenicsproject.org)
* multiphenics (https://github.com/multiphenics/multiphenics)

## Quick start

The easiest way to get started is to install FEniCS via Docker and then to install multiphenics and the EMIx package within the Docker container by following these steps.

### Downloading the EMIx package

Download and enter the EMIx source code

```
git clone git@github.com:pietrobe/EMIx.git
cd EMIx
```

### Installing FEniCS via Docker

To install FEniCS, we recommend using Docker (www.docker.com). EMIx relies on FEniCS (legacy), and has been tested most recently with the specific Docker image listed below. 

```
sudo docker run -t -v $(pwd):/home/fenics -i ghcr.io/scientificcomputing/fenics:2023-11-15
cd /home/fenics
```

### Installing multiphenics

The multiphenics package adds robust parallel multi-domain support to (legacy) FEniCS. Install multiphenics

```
pip install git+https://github.com/multiphenics/multiphenics.git
```

### Installing EMIx package

Install the EMIx package

```
pip install -e .
```

### Testing the installation

To test that all dependencies are operating successfully using a
sample EMIx problem, run

```
cd examples
python3 EMI_example.py                # Test serial run
mpirun -n N python3 -u EMI_example.py # Test parallel runs
```

The expected results are terminal output from 10 time steps of solving
the EMI equations and simulation results in a separate directory
(named output).

## Contributing guidelines

**EMIx** is developed by [Pietro Benedusi](https://pietrobe.github.io/) in collaboration with [Marie E. Rognes](https://marierognes.org/)'s group at [Simula](https://www.simula.no/).

## Miscellaneous

If you run into memory issues out-of-memory errors, check the memory limits of your Docker container, which may be set lower than the available system memory.

