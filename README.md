# EMIx
![EMIx](https://github.com/pietrobe/EMIx/docs/logos/EMIx_neg.png)
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

## Code

### Usage example (e.g. src/EMI/example.py)

Create dictionary with input files:

```
input_files = {'mesh_file':"path/mesh.xdmf", 'facets_file': "path/facets.xdmf", \
		'intra_restriction_dir': "path/interior_restriction.rtc.xdmf", \
		'extra_restriction_dir': "path/exterior_restriction.rtc.xdmf"}
```


* `mesh_file` contains volume tags
* `facets_file` contains surface tags
* `intra/extra_restriction_dir` contain multiphenics restrictions 

Read `data/README` for additional info about input generation from surface geometries.

Encode tag information in a dictionary:

```
tags = {'intra': 3 , 'extra': 1, 'boundary': 4, 'membrane': 2}
```

Construct (KNP)EMI problem given time step *dt*:

```
problem = EMI_problem(input_files, tags, dt)
```

Create ionic model (possibly for membrane subset, using tags=... argument, default is applied on all membranes) and call problem.init_ionic_model():

```
HH = HH_model(problem)	
problem.init_ionic_model([HH])
```


Create solver object, given *time_steps*, and solve

```
solver = EMI_solver(problem, time_steps)
solver.solve()
```


### Serial run 


`cd src/EMI`\
`python3 main.py`


###  Parallel run on N processors

`mpirun -n N python3 -u main.py`


### Structure and functionalities

The directories *src/KNP* and *src/EMI* contain files with the same structure, e.g. for EMI:

*src/EMI/EMI_problem.py*

* in the init() constructor mesh scaling factor and source factors can be set
* physical parameters, FEM order, initial and boundary conditions can be set, e.g.:


```
# physical parameters
C_M     = 0.01
sigma_i = 1.0
sigma_e = 1.0
	
# initial boundary potential 
phi_e_init = Constant(0.0)

# initial membrane potential 
phi_M_init = Constant(-0.06774) 

# order 
fem_order = 1

# BC
dirichlet_bcs = False
```


*src/EMI/EMI_solver.py*

* solver and output parameters can be set:

```
# solvers parameters, e.g. for EMI:
direct_solver  = False
ksp_rtol   	   = 1e-6
ksp_type   	   = 'cg'
pc_type    	   = 'hypre'
norm_type  	   = 'preconditioned'	
nonzero_init_guess = True 
verbose            = False

# output parameters	
save_xdmf_files = False
save_png_files  = False
save_mat        = False
```

*src/EMI/EMI_ionic_model.py*

* ionic model can be created and a membrane forcing factor set



##  Visualize output in Paraview
+ `Filters > Append Attributes` of both *solution.xdmf* and *subdomain.xdmf* (order is important to see time evolution)
+ `Filters > Threshold` according to subdomain tag and visualise field of interest


### Who do I talk to?

* Pietro Benedusi, Simula & USI, benedp@usi.ch
* Marie Rognes, Simula
* Halvor Herlyng, Simula