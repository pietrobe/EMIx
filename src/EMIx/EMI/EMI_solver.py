# Copyright © 2023 Pietro Benedusi
from EMIx.EMI.EMI_problem import EMI_problem
from EMIx.utils.misc      import dump
import numpy as np 
import matplotlib.pyplot as plt
import time
from dolfin       import *
from multiphenics import *
from petsc4py     import PETSc


class EMI_solver(object):

	# constructor
	def __init__(self, EMI_problem, time_steps, save_xdmf_files=False, save_png_files=False):
		
		self.problem    = EMI_problem
		self.time_steps = time_steps		

		# init forms		
		self.problem.setup_bilinear_form()	
		self.problem.setup_linear_form()	

		# output files		
		self.save_xdmf_files = save_xdmf_files
		self.save_png_files  = save_png_files
		
		if self.save_xdmf_files: self.init_xdmf()   		    		
		if self.save_png_files:  self.init_png()   		    		
		if self.save_mat:  self.time_steps = 1		

		# ininit ionic models 
		self.problem.init_ionic_model()	
		

	def assemble(self):	
		
		# alias
		p = self.problem
		
		if MPI.comm_world.rank == 0: print('Assembling linear system...') 	

		# assemble
		self.A = block_assemble(p.a)				

		if not self.direct_solver:
			# Iterative solver
			# Provide linear solver with PETSc structures 		
			self.A_ = as_backend_type(self.A).mat()											
			self.ksp.setOperators(self.A_, self.A_)

			if p.dirichlet_bcs or not self.set_nullspace:
				# Apply Dirichlet boundary conditions (BCs) to the linear system matrix 
				# Or pin for Neumann
				p.bcs.apply(self.A)
			else:
				# Pure Neumann BCs -> the system matrix is singular				
				# Handle system singularity by providing the nullspace of the linear system matrix 
				# to the linear solver.
				# Get the electric potential dofs in the restricted block function spaces
				ures2res_i = p.W.block_dofmap().original_to_block(0) # mapping unrestricted->restricted intra
				ures2res_e = p.W.block_dofmap().original_to_block(1) # mapping unrestricted->restricted extra

				# Get dofs of the potentials in the unrestricted spaces
				tot_num_dofs = p.W.sub(0).dofmap().index_map().local_range()[1]
				potential_dofs = list(range(0, tot_num_dofs))

				# Find the dofs of the potentials in the restricted spaces by 
				# indexing the unrestricted->restricted mapping with the dofs of the potentials
				# in the unrestricted spaces 
				res_phi_i_dofs = [ures2res_i[dof] for dof in potential_dofs if dof in ures2res_i]
				res_phi_e_dofs = [ures2res_e[dof] for dof in potential_dofs if dof in ures2res_e]
				
				# Create PETSc nullspace vector based on the structure of A
				ns_vec = self.A_.createVecLeft()

				# Set local values of nullspace vector and orthonormalize
				ns_vec.setValuesLocal(res_phi_i_dofs, np.array([1.0]*len(res_phi_i_dofs)))
				ns_vec.setValuesLocal(res_phi_e_dofs, np.array([1.0]*len(res_phi_e_dofs)))
				ns_vec.assemble()
				ns_vec.normalize()
				assert np.isclose(ns_vec.norm(), 1.0)

				# Create nullspace object
				self.nullspace = PETSc.NullSpace().create(vectors=[ns_vec], comm=MPI.comm_world)
				assert self.nullspace.test(self.A_)

				# Provide PETSc with the nullspace and orthogonalize the right-hand side vector
				# with respect to the nullspace
				as_backend_type(self.A_).setNullSpace(self.nullspace)
				as_backend_type(self.A_).setNearNullSpace(self.nullspace)
		else:
			# Direct solver
			if p.dirichlet_bcs:
				# Apply Dirichlet BCs to linear system matrix
				p.bcs.apply(self.A)
			
		if self.save_mat:
				
			print("Saving output/Amat...")  
			dump(self.A.mat(),'output/Amat')	
			exit()												
			
	def assemble_rhs(self):

		# alias
		p = self.problem		
			
		self.F = block_assemble(p.f)

		if p.dirichlet_bcs or not self.set_nullspace:
			# Problem either has 1. Dirichlet boundary conditions (BCs) on the domain boundary
			# or 2. pure Neumann BCs handled by a point Dirichlet BC
			# In both cases -> apply Dirichlet BCs			
			p.bcs.apply(self.F)

		if not self.direct_solver: 		
			self.F_ = as_backend_type(self.F).vec()	# TODO this only one?						
		
	def setup_solver(self):

		if self.direct_solver:
			
			if MPI.comm_world.rank == 0: print('Using direct solver...') 

		else:

			if MPI.comm_world.rank == 0: print('Setting up iterative solver...') 

			WH = self.problem.wh.block_vector()
			self.wh_ = as_backend_type(WH).vec()			
				
			self.ksp = PETSc.KSP().create()
			self.ksp.setType(self.ksp_type)
			pc = self.ksp.getPC()     
			pc.setType(self.pc_type)		
						
			PETScOptions.set("ksp_converged_reason")
			PETScOptions.set("ksp_rtol",      self.ksp_rtol)					
			PETScOptions.set("ksp_norm_type", self.norm_type)
			PETScOptions.set("ksp_initial_guess_nonzero",   self.nonzero_init_guess)						
						
			if self.problem.mesh.topology().dim() == 3: 
				PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.5)			

			if self.verbose:
				PETScOptions.set("ksp_view")
				PETScOptions.set("ksp_monitor_true_residual")
						
			self.ksp.setFromOptions() 

		# vectors to collect number of iterations and runtimes			
		self.iterations    = []
		self.solve_time    = []		
		self.assembly_time = []
		self.setup_time    = []
 

	def solve(self):

		# aliases		
		p      = self.problem
		t      = p.t
		dt     = p.dt
		wh     = p.wh		
		V      = p.V

		# setup
		self.setup_solver()	

		# assemble linear system matrix
		tic = time.perf_counter()		
		self.assemble()								
								
		if MPI.comm_world.rank == 0: print(f"Assembly in {time.perf_counter() - tic:0.4f} seconds")   			
		self.assembly_time.append(time.perf_counter() - tic)				
						
		# Time-stepping
		for i in range(self.time_steps):			

			# Update current time
			t.assign(float(t + dt))
			
			# print some infos
			if MPI.comm_world.rank == 0:
				print('\nTime step', i + 1) 							 
				print('t (ms) = ', 1000 * float(t)) 	

			# setup linear form
			tic = time.perf_counter()		
			p.setup_linear_form()	
			if MPI.comm_world.rank == 0: print(f"Setup linear form in {time.perf_counter() - tic:0.4f} seconds")   			
			self.setup_time.append(time.perf_counter() - tic)								
			
			# assemble rhs
			tic = time.perf_counter()		
			self.assemble_rhs()
								
			if MPI.comm_world.rank == 0: print(f"Time dependent assembly in {time.perf_counter() - tic:0.4f} seconds")   			
			self.assembly_time.append(time.perf_counter() - tic)						

			# Solve 		
			tic = time.perf_counter()

			if self.direct_solver:

				block_solve(self.A, wh.block_vector(), self.F, linear_solver = 'mumps')
				
			else:												

				# solve
				self.ksp.solve(self.F_, self.wh_)

				wh.block_vector().apply("")
				wh.apply("to subfunctions")   
						
			if MPI.comm_world.rank == 0: print(f"Solved in {time.perf_counter() - tic:0.4f} seconds")
			self.solve_time.append(time.perf_counter() - tic)	

			# update potential							
			p.phi_M.assign(wh[0] - wh[1])				

			####### write output #######
			
			if self.save_xdmf_files: self.save_xdmf()
			if self.save_png_files:  self.save_png()
				
			if not self.direct_solver:
				self.iterations.append(self.ksp.getIterationNumber())
							
			if i == self.time_steps - 1:
			
				total_assembly_time = sum(self.assembly_time)
				total_solve_time    = sum(self.solve_time)
				total_setup_time    = sum(self.setup_time)

				total_assembly_time = np.max(MPI.comm_world.gather(total_assembly_time,root=0))
				total_solve_time    = np.max(MPI.comm_world.gather(total_solve_time,   root=0))				
				total_setup_time    = np.max(MPI.comm_world.gather(total_setup_time,   root=0))				

				if MPI.comm_world.rank == 0: 										
					print("Assembly time:", total_assembly_time)
					print("Setup time:",    total_setup_time)
					print("Solve time:",    total_solve_time)

				# print solver and problem info
				self.print_info()		

				# close output files
				if self.save_xdmf_files: self.close_xdmf()
				if self.save_png_files:  self.plot_png()
		
	
	# print some infos
	def print_info(self):

		p = self.problem
		
		if MPI.comm_world.rank == 0:
			print("~~~~~~~~~~~~~~ Info ~~~~~~~~~~~~~~")
			print("MPI size =", MPI.size(MPI.comm_world))
			print("Input mesh =",          p.input_files['mesh_file'])
			print("Local mesh cells =",    p.mesh.num_cells())	
			print("Local mesh vertices =", p.mesh.num_vertices())				
			print("FEM order =", p.fem_order)

			if not self.direct_solver: print("System size =", self.A_.size[0])
			print("Time steps =",  self.time_steps)			
			print("dt =", float(p.dt))
			
			if p.dirichlet_bcs:
				print("Using Dirichlet bcs")
			else:
				print("Using Neumann bcs")

			print("Ionic models:")
			for model in p.ionic_models:
				print("-", model)

			print("~~~~~~~~~~~~~~ Solver ~~~~~~~~~~~~")
			if self.direct_solver:
				print("Direct solver: mumps")
			else:
				print('Type:', self.ksp_type,'+', self.pc_type)					
				print('Tolerance:', self.ksp_rtol)							
								
				print('Average iterations: ' + str(sum(self.iterations)/len(self.iterations)))				

			if self.save_xdmf_files: print('\nSaving XDMF files in output folder...')				
			if self.save_png_files:  print('\nSaving PNG files in output folder...\n')				


		

	def init_xdmf(self):

		# write tag data		
		xdmf_file = XDMFFile(MPI.comm_world, 'output/subdomains.xdmf')
		xdmf_file.write(self.problem.subdomains)	
		xdmf_file.close()			

		# write solution
		self.xdmf_file = XDMFFile(MPI.comm_world, "output/solution.xdmf")		

		self.xdmf_file.parameters['functions_share_mesh' ] = True
		self.xdmf_file.parameters['rewrite_function_mesh'] = False
		self.xdmf_file.parameters['flush_output']          = True	
		
		# set phi_e and phi_i just for visualization
		self.problem.wh[0].assign(interpolate(self.problem.phi_M_init, self.problem.V))			
		self.problem.wh[1].assign(interpolate(self.problem.phi_e_init, self.problem.V))

		self.xdmf_file.write(self.problem.wh[0], float(self.problem.t))		
		self.xdmf_file.write(self.problem.wh[1], float(self.problem.t))		
			


	def save_xdmf(self):
		
		self.xdmf_file.write(self.problem.wh[0], float(self.problem.t))		
		self.xdmf_file.write(self.problem.wh[1], float(self.problem.t))		
		

	def close_xdmf(self):

		self.xdmf_file.close()		

	
	def init_png(self):

		p = self.problem

		self.point_to_plot = []		

		# for gamma point
		f_to_v = p.mesh.topology()(p.mesh.topology().dim()-1, 0)
		dmap   = p.V.dofmap()			

		# loop over facets updating gating only on gamma
		for facet in facets(p.mesh):

			if p.boundaries.array()[facet.index()] in p.gamma_tags:

				vertices = f_to_v(facet.index())

				local_indices = dmap.entity_closure_dofs(p.mesh, 0, [vertices[0]])				

				self.point_to_plot = local_indices	

				break											
				
		# prepare data structures		
		imap = dmap.index_map()
		num_dofs_local = imap.size(IndexMap.MapSize.ALL) * imap.block_size()
		
		local_phi = p.phi_M.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))			
		
		# potential
		self.v_t = []
		self.v_t.append(1000 *local_phi[self.point_to_plot]) # converting to mV 		
		self.out_v_string = 'output/v.png'					


	def save_png(self):
		
		p = self.problem

		# prepare data (needed for parallel)
		dmap = p.V.dofmap()		
		imap = dmap.index_map()
		num_dofs_local = imap.size(IndexMap.MapSize.ALL) * imap.block_size()

		local_phi = p.phi_M.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))			

		self.v_t.append(1000 * local_phi[self.point_to_plot]) # converting to mV 

	def plot_png(self):
		
		# aliases
		dt = float(self.problem.dt)
		time_steps = self.time_steps

		# save plot of membrane potential
		plt.figure(0)
		plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps + 1), self.v_t)
		plt.xlabel('time (ms)')
		plt.ylabel('membrane potential (mV)')
		plt.savefig(self.out_v_string)


	# solvers parameters
	direct_solver  = False
	ksp_rtol   	   = 1e-6
	ksp_type   	   = 'cg'
	pc_type    	   = 'hypre'
	norm_type  	   = 'preconditioned'	
	nonzero_init_guess = True 
	verbose            = False
	
	# output parameters	
	save_mat        = False

	# handling pure Neumann boundary conditions
	set_nullspace = False  # True = provide linear solver with the nullspace of the system matrix,
            			   # False = pin the solution with a point Dirichlet BC
