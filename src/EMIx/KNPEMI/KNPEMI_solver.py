# Copyright © 2023 Pietro Benedusi
from EMIx.KNPEMI.KNPEMI_problem import KNPEMI_problem 
from EMIx.utils.misc            import norm_2, dump, get_adaptive_dt
import numpy as np 
import matplotlib.pyplot as plt
import time
from dolfin       import *
from multiphenics import *
from petsc4py     import PETSc


class KNPEMI_solver(object):

	# constructor
	def __init__(self, KNPEMI_problem, save_xdmf_files=False, save_png_files=False, save_mat=False):

		# init variables 
		self.problem    = KNPEMI_problem		
		self.save_xdmfs = save_xdmf_files	
		self.save_pngs  = save_png_files	
		self.save_mat   = save_mat	

		# init variational form
		self.problem.setup_variational_form()				
			
		# output files		
		if self.save_xdmfs: self.init_xdmf_savefile()
		if self.save_pngs:  self.init_png_savefile()	
		
		# perform a single time step when saving matrices
		if self.save_mat: self.problem.time_steps = 1 
		
		if self.problem.MMS_test: self.problem.print_errors()		

		# ininit ionic models 
		self.problem.init_ionic_model()					


	def assemble(self, init=True):	
		
		# alias
		p = self.problem
		
		if MPI.comm_world.rank == 0: print('Assembling linear system...') 	

		if init:				
			
			if MPI.comm_world.rank == 0 and init: print('Init matrices and KSP operators...') 	

			self.A = block_assemble(p.a)				
			self.F = block_assemble(p.L)

			# create PETSC data structures for the iterative solver
			if not self.direct_solver: 							

				self.A_ = as_backend_type(self.A).mat()
				self.F_ = as_backend_type(self.F).vec()							
								
				if self.use_P_mat:
					self.ksp.setOperators(self.A_, self.P_) 
				else:
					self.ksp.setOperators(self.A_, self.A_)		

			# impose BC
			if p.dirichlet_bcs:

				# Apply Dirichlet boundary conditions and the right-hand side vector				
				p.bcs.apply(self.A)
				p.bcs.apply(self.F)

			# Neumann BC (direct solver does it in automatic)
			elif not self.direct_solver:

				# Handle system singularity providing a nullspace
				if self.set_nullspace:					

					# Get the electric potential dofs in the restricted block function spaces
					ures2res_i = p.W.block_dofmap().original_to_block(0) # mapping unrestricted->restricted intra
					ures2res_e = p.W.block_dofmap().original_to_block(1) # mapping unrestricted->restricted extra

					# Get dofs of the potentials in the unrestricted spaces
					tot_num_dofs = p.W.sub(0).dofmap().index_map().local_range()[1]
					potential_dofs = list(range(p.N_ions, tot_num_dofs, p.N_ions+1))

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
					nullspace = PETSc.NullSpace().create(vectors=[ns_vec], comm=MPI.comm_world)
					assert nullspace.test(self.A_)

					# Provide PETSc with the nullspace and orthogonalize the right-hand side vector
					# with respect to the nullspace
					as_backend_type(self.A_).setNullSpace(nullspace)
					as_backend_type(self.A_).setNearNullSpace(nullspace)

				else:
					# Handle system singularity by pinning potential using a point Dirichlet BC
					p.bcs.apply(self.A)
					p.bcs.apply(self.F)						
		else:

			# Assemble the linear system matrix A if reassembly is enabled
			if self.reassemble_A: 
				block_assemble(p.a, block_tensor=self.A)				
			
				if p.dirichlet_bcs or not self.set_nullspace:					
					p.bcs.apply(self.A)			
	
			else:
				if MPI.comm_world.rank == 0: print("Skipping matrix A assembly.")
			
			# Assemble the right-hand side
			block_assemble(p.L, block_tensor=self.F)		

			if p.dirichlet_bcs or not self.set_nullspace:				
				p.bcs.apply(self.F)			
	
	def assemble_preconditioner(self):				

		self.P = block_assemble(self.problem.P)
		if self.problem.dirichlet_bcs or not self.set_nullspace:
			# Problem either has 1. Dirichlet boundary conditions (BCs) on the domain boundary
			# or 2. pure Neumann BCs handled by a point Dirichlet BC
			# In both cases -> apply Dirichlet BCs	
			self.problem.bcs.apply(self.P)

		self.P_ = as_backend_type(self.P).mat()

		if self.save_mat: 
			if self.problem.MMS_test:
				print("Saving Pmat_MMS...")  
				dump(self.P.mat(),'output/Pmat_MMS')				
			else:
				print("Saving output/Pmat")  
				dump(self.P.mat(),'output/Pmat')				

	
	def setup_solver(self):

		if self.direct_solver:
			
			if MPI.comm_world.rank == 0: print('Using direct solver...') 

		else:

			p = self.problem

			if MPI.comm_world.rank == 0: print('Setting up iterative solver...') 

			# set initial guess						
			for idx, ion in enumerate(p.ion_list):				
														
				assign(p.wh[0].sub(idx), interpolate(ion['ki_init'], p.V.sub(idx).collapse()))
				assign(p.wh[1].sub(idx), interpolate(ion['ke_init'], p.V.sub(idx).collapse()))
			
			assign(p.wh[0].sub(p.N_ions), interpolate(p.phi_i_init, p.V.sub(idx).collapse()))
			assign(p.wh[1].sub(p.N_ions), interpolate(p.phi_e_init, p.V.sub(idx).collapse()))
					

			WH = p.wh.block_vector()
			self.wh_ = as_backend_type(WH).vec()						
						
			self.ksp = PETSc.KSP().create()
			self.ksp.setType(self.ksp_type)
			pc = self.ksp.getPC()     
			pc.setType(self.pc_type)		

			if self.pc_type == "fieldsplit":

				# alias 
				Wi = self.problem.W.sub(0)
				We = self.problem.W.sub(1)

				is0 = PETSc.IS().createGeneral(Wi.sub(0).dofmap().dofs())
				is1 = PETSc.IS().createGeneral(Wi.sub(1).dofmap().dofs())
				is2 = PETSc.IS().createGeneral(Wi.sub(2).dofmap().dofs())
				is3 = PETSc.IS().createGeneral(Wi.sub(3).dofmap().dofs())
				is4 = PETSc.IS().createGeneral(We.sub(0).dofmap().dofs())
				is5 = PETSc.IS().createGeneral(We.sub(1).dofmap().dofs())
				is6 = PETSc.IS().createGeneral(We.sub(2).dofmap().dofs())
				is7 = PETSc.IS().createGeneral(We.sub(3).dofmap().dofs())

				fields = [('0', is0), ('1', is1),('2', is2), ('3', is3),('4', is4), ('5', is5),('6', is6), ('7', is7)]
				pc.setFieldSplitIS(*fields)
			
				ksp_solver = 'preonly'
				P_inv      = 'hypre'
				
				PETScOptions.set('pc_fieldsplit_type', 'additive')
				
				PETScOptions.set('fieldsplit_0_ksp_type', ksp_solver )
				PETScOptions.set('fieldsplit_1_ksp_type', ksp_solver)
				PETScOptions.set('fieldsplit_2_ksp_type', ksp_solver)
				PETScOptions.set('fieldsplit_3_ksp_type', ksp_solver)
				PETScOptions.set('fieldsplit_4_ksp_type', ksp_solver)
				PETScOptions.set('fieldsplit_5_ksp_type', ksp_solver)
				PETScOptions.set('fieldsplit_6_ksp_type', ksp_solver)
				PETScOptions.set('fieldsplit_7_ksp_type', ksp_solver)

				PETScOptions.set('fieldsplit_0_pc_type',  P_inv)				
				PETScOptions.set('fieldsplit_1_pc_type',  P_inv)
				PETScOptions.set('fieldsplit_2_pc_type',  P_inv)
				PETScOptions.set('fieldsplit_3_pc_type',  P_inv)
				PETScOptions.set('fieldsplit_4_pc_type',  P_inv)
				PETScOptions.set('fieldsplit_5_pc_type',  P_inv)
				PETScOptions.set('fieldsplit_6_pc_type',  P_inv)
				PETScOptions.set('fieldsplit_7_pc_type',  P_inv)

				
			PETScOptions.set("ksp_converged_reason")
			PETScOptions.set("ksp_rtol",      self.ksp_rtol)		
			PETScOptions.set("ksp_max_it",    self.ksp_max_it)		
			PETScOptions.set("ksp_norm_type", self.norm_type)
			PETScOptions.set("ksp_initial_guess_nonzero",   self.nonzero_init_guess)
			PETScOptions.set("pc_hypre_boomeramg_max_iter", self.max_amg_iter)				
			PETScOptions.set("pc_factor_zeropivot", 1e-22)		
			
			# PETScOptions.set("pc_factor_mat_solver_type",'umfpack')	
			# PETScOptions.set("ksp_gmres_modifiedgramschmidt")	
			# PETScOptions.set("pc_hypre_boomeramg_print_statistics")	
			# PETScOptions.set("pc_hypre_boomeramg_grid_sweeps_all", 3)
			# PETScOptions.set("pc_hypre_boomeramg_coarsen_type", "HMIS")									

			if p.mesh.topology().dim() == 3: 
				PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.5)
				#PETScOptions.set("pc_hypre_boomeramg_agg_nl", 6) # TODO use for some gain 

			# if self.ksp_type == 'fgmres': PETScOptions.set("ksp_gmres_restart", self.gmres_restart)				

			if self.verbose:
				PETScOptions.set("ksp_view")
				PETScOptions.set("ksp_monitor_true_residual")
						
			self.ksp.setFromOptions() 

		# vectors to collect number of iterations and runtimes			
		self.iterations    = []
		self.solve_time    = []		
		self.assembly_time = []
 

	def solve(self):

		# setup
		self.setup_solver()		
		
		# aliases		
		p      = self.problem
		t      = p.t		
		wh     = p.wh
		N_ions = p.N_ions				
		V      = p.V.sub(N_ions).collapse()

		# assemble preconditioner if needed
		if not self.direct_solver and self.use_P_mat: 
			self.problem.setup_preconditioner(self.use_block_Jacobi)		
			self.assemble_preconditioner()		
				
		setup_timer = 0		

		self.t_list  = []		
		self.t_list.append(float(t))
		if self.time_adaptive: self.dt_list = []

		# Time-stepping
		for i in range(p.time_steps):			

			dt = p.dt

			# Update current time
			t.assign(float(t + dt))
			self.t_list.append(float(t))
			if self.time_adaptive: self.dt_list.append(1000*float(dt))

			# print some infos
			if MPI.comm_world.rank == 0:
				print('\nTime step', i + 1) 							 
				print('t (ms) = ', 1000 * float(t)) 	

			# set if assembly A						
			self.reassemble_A = (i % self.assembly_interval == 0)				
			
			tic = time.perf_counter()	
			p.setup_variational_form()		
			setup_timer += time.perf_counter() - tic						

			# assemble	    
			tic = time.perf_counter()		
			self.assemble(i==0)											
								
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
			
			if not self.direct_solver:
				
				its = self.ksp.getIterationNumber()
				self.iterations.append(its)

				# use time adaptivity
				if self.time_adaptive:
					p.dt = Constant(get_adaptive_dt(float(dt), its))				

			if self.save_mat:
				
				if self.problem.MMS_test:
					print("Saving Amat_MMS...")  
					dump(self.A.mat(),'output/Amat_MMS')				
				else:
					print("Saving output/Amat...")  
					dump(self.A.mat(),'output/Amat')				
			
				# use then in MATLAB: data = readNPY('Amat.npy'); A = create_sparse_mat_from_data(data);
				# Write b
				# out_string = 'output/bvec.m'
				# original_stdout = sys.stdout
				# np.set_printoptions(threshold=sys.maxsize, linewidth=1000000000)
				# with open(out_string, 'w') as f:
				# 	sys.stdout = f
				# 	print("b = ", self.F.get_local(), ";")
				# 	sys.stdout = original_stdout
				# print("b = ", self.F.get_local(), ";")

				# with open('output/wvec.m', 'w') as f:
				# 	sys.stdout = f
				# 	print("w = ", w.get_local(), ";")
				# 	sys.stdout = original_stdout # Reset the standard output to its original value

			# update previous solution
			p.u_p.sub(0).assign(wh[0])
			p.u_p.sub(1).assign(wh[1])
					
			# update membrane potential
			p.phi_M_prev.assign(interpolate(wh[0].sub(N_ions), V) - interpolate(wh[1].sub(N_ions), V))						

			# write output
			if self.save_xdmfs and (i % self.save_interval == 0) : self.save_xdmf()		
			if self.save_pngs:	self.save_png()				
							
			if i == p.time_steps - 1:
							
				total_assembly_time = sum(self.assembly_time)
				total_solve_time    = sum(self.solve_time)

				total_assembly_time = np.max(MPI.comm_world.gather(total_assembly_time,root=0))
				total_solve_time    = np.max(MPI.comm_world.gather(total_solve_time,   root=0))
				setup_timer         = np.max(MPI.comm_world.gather(setup_timer,        root=0))

				if MPI.comm_world.rank == 0: 
					print("\nTotal setup time:",  setup_timer)
					print("Total assemble time:", total_assembly_time)
					print("Total solve time:",    total_solve_time)

				# print solver and problem info
				self.print_info()

				if self.save_pngs:  
					self.print_figures()	
					if MPI.comm_world.rank == 0: print("\nPNG output saved in", self.out_file_prefix)
				
				if self.save_xdmfs: 
					self.close_xdmf()	
					if MPI.comm_world.rank == 0: print("\nXDMF output saved in", self.out_file_prefix)									

		
	
	# print some infos
	def print_info(self):

		p = self.problem
		
		if MPI.comm_world.rank == 0:
			print("~~~~~~~~~~~~~~ Info ~~~~~~~~~~~~~~")
			print("MPI size =", MPI.size(MPI.comm_world))
			print("Input mesh =", p.input_files['mesh_file'])
			print("Local mesh cells =", p.mesh.num_cells())	
			print("Local mesh vertices =", p.mesh.num_vertices())				
			print("FEM order =", p.fem_order)

			if not self.direct_solver: print("System size =", self.A_.size[0])
			print("Time steps =",  p.time_steps)			
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
				
				if self.use_P_mat: 						
						print('Enabling preconditioning')

				print('Average iterations: ' + str(sum(self.iterations)/len(self.iterations)))				
		

	def init_png_savefile(self):

		p = self.problem

		self.point_to_plot = []		

		# for gamma point
		f_to_v = p.mesh.topology()(p.mesh.topology().dim()-1, 0)
		dmap   = p.V.sub(p.N_ions).collapse().dofmap()			

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
		
		local_phi = p.phi_M_prev.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))			
		
		# potential
		self.v_t = []
		self.v_t.append(1000 *local_phi[self.point_to_plot]) # converting to mV 		
		self.out_v_string = self.out_file_prefix + 'v.png'					

		# # concentrations TEST
		# ui_p = self.problem.u_p.sub(0)		
		# local_Na = ui_p.sub(0).vector().get_local(np.arange(num_dofs_local, dtype=np.int32))	

		# self.Na_t = []
		# self.Na_t.append(local_Na[-1]) # converting to mV 		
		# self.out_Na_string = self.out_file_prefix + 'Na_t.png'					
		
		# if HH gating variables are present (TODO move)
		if hasattr(p, 'n'):

			local_n = p.n.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
			local_m = p.m.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
			local_h = p.h.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
			
			self.n_t = []
			self.m_t = []
			self.h_t = []
			
			self.n_t.append(local_n[self.point_to_plot]) 
			self.m_t.append(local_m[self.point_to_plot]) 
			self.h_t.append(local_h[self.point_to_plot]) 
			
			self.out_gate_string =  self.out_file_prefix + 'gating.png'
			

	# methods for output
	def save_png(self):
		
		p = self.problem

		# prepare data (needed for parallel)
		dmap = p.V.sub(p.N_ions).collapse().dofmap()		
		imap = dmap.index_map()
		num_dofs_local = imap.size(IndexMap.MapSize.ALL) * imap.block_size()

		local_phi = p.phi_M_prev.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))			

		self.v_t.append(1000 * local_phi[self.point_to_plot]) # converting to mV 		

		if hasattr(p, 'n'):
			
			local_n = p.n.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
			local_m = p.m.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
			local_h = p.h.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
						
			self.n_t.append(local_n[self.point_to_plot]) 
			self.m_t.append(local_m[self.point_to_plot]) 
			self.h_t.append(local_h[self.point_to_plot]) 


	def print_figures(self):

		# aliases
		dt = float(self.problem.dt)
		time_steps = self.problem.time_steps

		# save plot of membrane potential
		plt.figure(0)
		plt.plot(self.t_list, self.v_t)
		plt.xlabel('time (ms)')
		plt.ylabel('membrane potential (mV)')
		plt.savefig(self.out_v_string)

		# save plot of gating variables
		if hasattr(self.problem, 'n'):
			plt.figure(1)
			plt.plot(self.t_list, self.n_t, label='n')
			plt.plot(self.t_list, self.m_t, label='m')
			plt.plot(self.t_list, self.h_t, label='h')
			plt.legend()
			plt.xlabel('time (ms)')
			plt.savefig(self.out_gate_string)

		# save iteration history
		if not self.direct_solver:
			plt.figure(2)
			plt.plot(self.iterations)
			plt.xlabel('time step')
			plt.ylabel('number of iterations')
			plt.savefig(self.out_file_prefix + 'iterations.png')

		if self.time_adaptive:
				plt.figure(3)
				plt.plot(self.dt_list)
				plt.xlabel('time step')
				plt.ylabel('dt (ms)')
				plt.savefig(self.out_file_prefix + 'dt.png')
				
		# save runtime data
		plt.figure(4)
		plt.plot(self.assembly_time, label='assembly')
		plt.plot(self.solve_time, label='solve')
		plt.legend()
		plt.xlabel('time step')
		plt.ylabel('Time (s)')
		plt.savefig(self.out_file_prefix + 'timings.png')

	
	def init_xdmf_savefile(self):

		# write tag data
		filename  = self.out_file_prefix + 'subdomains.xdmf'
		xdmf_file = XDMFFile(MPI.comm_world, filename)
		xdmf_file.write(self.problem.subdomains)	
		xdmf_file.close()	
		
		# write solution
		ui_p = self.problem.u_p.sub(0)
		ue_p = self.problem.u_p.sub(1)

		filename = self.out_file_prefix + 'solution.xdmf'
		
		self.xdmf_file = XDMFFile(MPI.comm_world, filename)		
		self.xdmf_file.parameters['rewrite_function_mesh'] = False
		self.xdmf_file.parameters['functions_share_mesh']  = True
		self.xdmf_file.parameters['flush_output']          = True		
		
		for idx in range(self.problem.N_ions + 1):
			
			self.xdmf_file.write(ui_p.sub(idx), float(self.problem.t))						
			self.xdmf_file.write(ue_p.sub(idx), float(self.problem.t))		


		if self.save_fluxes:

			filename = self.out_file_prefix + 'fluxes.xdmf'			

			self.xdmf_flux = XDMFFile(MPI.comm_world, filename)			

			self.xdmf_flux.parameters['rewrite_function_mesh'] = False
			self.xdmf_flux.parameters['functions_share_mesh']  = True
			self.xdmf_flux.parameters['flush_output']          = True
			
			# DG0 space for plotting
			self.V_DG0 = FunctionSpace(self.problem.mesh, "DG", 0)								

		return


	def save_xdmf(self):

		ui_p = self.problem.u_p.sub(0)
		ue_p = self.problem.u_p.sub(1)
		
		for idx in range(self.problem.N_ions + 1):

			self.xdmf_file.write(ui_p.sub(idx), float(self.problem.t))						
			self.xdmf_file.write(ue_p.sub(idx), float(self.problem.t))						
		
		if self.save_fluxes:	

			phi_e_prev = ue_p[self.problem.N_ions] 
			phi_i_prev = ui_p[self.problem.N_ions]	

			# Setup ion specific part of variational formulation
			for idx, ion in enumerate(self.problem.ion_list):
				
				# get ion attributes
				z  = ion['z' ];
				Di = ion['Di'];
				De = ion['De'];			
				psi = self.problem.psi

				ki_prev = ui_p[idx] 
				ke_prev = ue_p[idx] 
				
				J_i_diff  = grad(Di*ki_prev)
				J_i_drift = Constant(Di*z/psi)*ki_prev*grad(phi_i_prev)

				J_e_diff  = grad(De*ke_prev)
				J_e_drift = Constant(De*z/psi)*ke_prev*grad(phi_e_prev)
				
				diff_i  = project(norm_2(J_i_diff),  self.V_DG0)
				diff_e  = project(norm_2(J_e_diff),  self.V_DG0)
				drift_i = project(norm_2(J_i_drift), self.V_DG0)
				drift_e = project(norm_2(J_e_drift), self.V_DG0)				
										
				diff_i.rename('J_diff_i', '')				
				diff_e.rename('J_diff_e', '')				

				self.xdmf_flux.write(diff_i, float(self.problem.t))
				self.xdmf_flux.write(diff_e, float(self.problem.t))				

				drift_i.rename('J_drift_i', '')				
				drift_e.rename('J_drift_e', '')								
												
				self.xdmf_flux.write(drift_i, float(self.problem.t))
				self.xdmf_flux.write(drift_e, float(self.problem.t))				

		return

	
	def close_xdmf(self):

		self.xdmf_file.close()		
		
		if self.save_fluxes:
			self.xdmf_flux.close()			

		return	
	
	# solvers parameters
	direct_solver  	   = False
	ksp_rtol   	   	   = 1e-6
	ksp_max_it     	   = 1000
	ksp_type   	   	   = 'gmres'
	pc_type    	       = 'hypre'
	norm_type  	       = 'preconditioned'
	max_amg_iter       = 1
	use_P_mat          = True
	use_block_Jacobi   = True
	nonzero_init_guess = True 

	# misc.
	assembly_interval = 1  # with 1 assemble matrix each A for each time step
	verbose           = False	
	time_adaptive     = False

	# handling pure Neumann boundary conditions
	set_nullspace = False  # True = provide linear solver with the nullspace of the system matrix,
						  # False = pin the solution with a point Dirichlet BC

	# output parameters
	out_file_prefix = 'output/'
	save_interval = 1	
	save_fluxes = False