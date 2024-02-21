<img src="./docs/logos/EMIx.png" width="300" height="150">

Framework for parallel simulations of EMI and KNP-EMI models in complex geometries

## Dependencies
* FEniCS
* multiphenics

### Setup FEniCS docker


```
docker run -t -v $(pwd):/home/fenics -i ghcr.io/scientificcomputing/fenics:2023-11-15
cd /home/fenics
```

### Install multiphenics
```
pip install git+https://github.com/multiphenics/multiphenics.git
```

### Install EMIx package
```
pip install -e .
```

### Serial run 

`cd src/examples`\
`python3 EMI_example.py`


###  Parallel run on N processors

`mpirun -n N python3 -u EMI_example.py`


**EMIx** is developed by [Pietro Benedusi](https://pietrobe.github.io/) in collaboration with [Marie E. Rognes](https://marierognes.org/)'s group at [Simula](https://www.simula.no/).
