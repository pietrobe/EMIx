### EMI usage, based on *EMI_example.py*

Given an input file config.yml create an EMI (or KNPEMI) problem
```
problem = EMI_problem("config.yml")
```
The input .yml file contains time discretisation parameters, input files, tags information, and physical parameters. If not specified in config.yml, default input parameters are taken from (KNP)EMI_problem.py

Remark on tags: 
- when pure Neumann boundary conditions are used (default) the boundary tag is not necessary. 
- The default value of the extra tag is 1 and the default of membrane is the same as the intra tag. 
- The intra tag must be provided (as an integer or a list). 

Read `data/README.md` for additional info about input generation from surface geometries.

Create ionic model (possibly for membrane subset, using tags=... argument, default is applied on all membranes) and call problem.init_ionic_model():

```
HH = HH_model(problem)	
problem.init_ionic_model([HH])
```


Create solver object and solve

```
solver = EMI_solver(problem)
solver.solve()
```

### Structure and functionalities

*EMI_solver.py*

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
save_mat        = False
```

*EMI_ionic_model.py*

* new ionic model can be created


##  Visualize output in Paraview
+ `Filters > Append Attributes` of both *solution.xdmf* and *subdomain.xdmf* (order is important to see time evolution)
+ `Filters > Threshold` according to subdomain tag and visualise field of interest
