<img src="./docs/logos/EMIx.png" width="300" height="150">

Framework for parallel simulations of EMI and KNP-EMI models in complex geometries. 

## Dependencies

* FEniCS (www.fenicsproject.org)
* multiphenics (https://github.com/multiphenics/multiphenics)

## Installation 

The quickest way to get started is to install and run FEniCS via Docker and then to install multiphenics and the EMIx package within the Docker container. The steps are then to

### Downloading the EMIx package

To download and enter the EMIx source code

```
git clone git@github.com:pietrobe/EMIx.git
cd EMIx
```

### Installing FEniCS via Docker

To (install and) run FEniCS, we recommend using Docker (www.docker.com). EMIx relies on FEniCS (legacy), and has been tested with the Docker image listed below:

```
sudo docker run -t -v $(pwd):/home/fenics -i ghcr.io/scientificcomputing/fenics:2023-11-15
cd /home/fenics
```

Running FEniCS via Docker with default memory limits is suitable for small to moderately sized problems.

### Install multiphenics
```
pip install git+https://github.com/multiphenics/multiphenics.git
```

### Install EMIx package
```
pip install -e .
```

### Testing the installation

To test that all dependencies are operating successfully and to run a
sample EMIx problem, we recommend:

```
cd examples
python3 EMI_example.py                # In serial
mpirun -n N python3 -u EMI_example.py # Test parallel runs
```

The expected result is terminal output from 10 time steps of solving
the EMI equations and solver output in an output directory.


## Contributing guidelines

**EMIx** is developed by [Pietro Benedusi](https://pietrobe.github.io/) in collaboration with [Marie E. Rognes](https://marierognes.org/)'s group at [Simula](https://www.simula.no/).
