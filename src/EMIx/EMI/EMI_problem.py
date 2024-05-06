# Copyright © 2023 Pietro Benedusi
from EMIx.utils.Mix_dim_problem  import Mixed_dimensional_problem
from EMIx.EMI.EMI_ionic_model import *
from dolfin        import *
from multiphenics  import *
import numpy as np
import time

# required by dS
parameters["ghost_mode"] = "shared_facet" 

# optimizaion flags
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3  -ffast-math'

parameters["form_compiler"]["quadrature_degree"] = 5 # TEST

# parameters["reorder_dofs_serial"] = False # TEST: False = paper ordering


class EMI_problem(Mixed_dimensional_problem):
								
	def init(self):
		
		pass

	def	add_ionic_model(self, model_type, tags=None, stim_fun=g_syn_none):

		if model_type == "HH":			

			model = HH_model(self, tags, stim_fun);
			self.ionic_models.append(model)	

		elif model_type == "Passive":			
			
			model = Passive_model(self, tags);
			self.ionic_models.append(model)	

		else:
			print("ERROR: ", model_type, " not supported")
			exit()

		
	def setup_spaces(self):

		if MPI.comm_world.rank == 0: print('Creating spaces...') 
			
		self.V = FunctionSpace(self.mesh, "Lagrange", self.fem_order)

		self.W = BlockFunctionSpace([self.V, self.V], restrict=[self.interior, self.exterior])				

		# create function for solution
		self.wh  = BlockFunction(self.W)
		
		# rename for more readable output		
		self.wh[0].rename('phi_i', '')
		self.wh[1].rename('phi_e', '')		

							
	def setup_boundary_conditions(self):

		# alias 		
		We = self.W.sub(1)

		# add Dirichlet boundary conditions on exterior boundary		
		bce = []
		
		if self.dirichlet_bcs: 
						
			bc = DirichletBC(We, self.phi_e_init, self.boundaries, self.bound_tag)
			bce.append(bc)					
					
			self.bcs = BlockDirichletBC([None, bce]) 
		
		else: # set point-wise BC for natural BC			
			
			for v in vertices(self.mesh):

				self.x0 = v.point().x()
				self.y0 = v.point().y()
				self.z0 = v.point().z()							

				break

			# broadcast from rank = 0 (same point for all procs)
			self.x0 = MPI.comm_world.bcast(self.x0, root=0)	
			self.y0 = MPI.comm_world.bcast(self.y0, root=0)	
			self.z0 = MPI.comm_world.bcast(self.z0, root=0)	
			
			def point_bc(x, on_boundary):
				tol = DOLFIN_EPS
				
				# 2D
				if self.mesh.topology().dim() == 2:

					return (abs(x[0] - self.x0) < tol) and (abs(x[1] - self.y0) < tol) 
				# 3D
				else:

					return (abs(x[0] - self.x0) < tol) and (abs(x[1] - self.y0) < tol) and (abs(x[2] - self.z0) < tol)
		
			bc = DirichletBC(We, self.phi_e_init, point_bc, method="pointwise")
			bce.append(bc)		

			self.bcs = BlockDirichletBC([None, bce])

	
	def setup_bilinear_form(self):

		# sanity check
		if len(self.ionic_models) == 0 and MPI.comm_world.rank == 0:		
			print('\nERROR: call init_ionic_model() to provide ionic models!\n')
			exit()

		# aliases
		dt       = self.dt 
		C_M      = self.C_M
		sigma_i  = self.sigma_i
		sigma_e  = self.sigma_e	
					
		if MPI.comm_world.rank == 0: print('Setting up bilinear form...') 

		# define measures
		dx = Measure("dx")(subdomain_data=self.subdomains)
		dS = Measure("dS")(subdomain_data=self.boundaries)

		dxi = dx(self.intra_tags)
		dxe = dx(self.extra_tag)			
		dS  = dS(self.gamma_tags) 
						
		# trial/test functions
		uu = BlockTrialFunction(self.W)
		vv = BlockTestFunction(self.W)

		(ui, ue) = block_split(uu)
		(vi, ve) = block_split(vv)

		a11 = inner(sigma_i*grad(ui), grad(vi))*dxi + (C_M/dt) * inner(ui('+'), vi('+'))*dS
		a22 = inner(sigma_e*grad(ue), grad(ve))*dxe + (C_M/dt) * inner(ue('+'), ve('+'))*dS
		a12 = - (C_M/dt) * inner(ue('+'), vi('+'))*dS
		a21 = - (C_M/dt) * inner(ui('+'), ve('+'))*dS

		
		self.a = [[a11, a12],
				 [ a21, a22]]		


	def setup_preconditioner(self):
		
		# aliases
		dt       = self.dt 
		C_M      = self.C_M
		sigma_i  = self.sigma_i
		sigma_e  = self.sigma_e	
					
		if MPI.comm_world.rank == 0: print('Setting up preconditioner') 

		# define measures
		dx = Measure("dx")(subdomain_data=self.subdomains)		
		dS = Measure("dS")(subdomain_data=self.boundaries)

		dxi = dx(self.intra_tags)
		dxe = dx(self.extra_tag)					
		dS  = dS(self.gamma_tags) 
						
		# trial/test functions
		uu = BlockTrialFunction(self.W)
		vv = BlockTestFunction(self.W)

		(ui, ue) = block_split(uu)
		(vi, ve) = block_split(vv)

		p11 = inner(sigma_i*grad(ui), grad(vi))*dxi + inner(ui, vi)*dxi
		p22 = inner(sigma_e*grad(ue), grad(ve))*dxe + inner(ue, ve)*dxe
		
		self.prec = [[p11, 0],
				    [ 0, p22]]		

		# # setup BC for preconditioner		
		# bc_i = DirichletBC(self.W.sub(0), self.phi_M_init, self.boundaries, self.gamma_tags[0])	
		# bc_e = DirichletBC(self.W.sub(1), self.phi_e_init, self.boundaries, self.bound_tag)																					
		# self.bcs_prec = BlockDirichletBC([bc_i, None]) 		


	def setup_linear_form(self): # remove t from input
		
		# aliases
		dt       = self.dt 
		C_M      = self.C_M		
		source_i = self.source_i
		source_e = self.source_e
		t        = float(self.t)				

		if np.isclose(t,0): self.phi_M = interpolate(self.phi_M_init, self.V)                  

		# update source term (Needed?)
		# source_i.t = self.t   
						
		if MPI.comm_world.rank == 0: print('Setting up linear form...') 

		# define measures
		dx = Measure("dx")(subdomain_data=self.subdomains)
		dS = Measure("dS")(subdomain_data=self.boundaries)

		dxi = dx(self.intra_tags)
		dxe = dx(self.extra_tag)			
		dS  = dS(self.gamma_tags) 
						
		# trial/test functions		
		vv = BlockTestFunction(self.W)		
		(vi, ve) = block_split(vv)

		# insert source terms
		fi = inner(source_i, vi)*dxi
		fe = inner(source_e, ve)*dxe

		# init dictionary for ionic channel
		I_ch = dict.fromkeys(self.gamma_tags) 			
						
		# loop over ionic models
		for model in self.ionic_models:										

			# loop over ionic model tags
			for gamma_tag in model.tags:	
										
				I_ch[gamma_tag] = model._eval()						


		for gamma_tag in self.gamma_tags:

			fg = self.phi_M - (dt/C_M) * I_ch[gamma_tag]						

			fi += (C_M/dt) * inner(fg, vi('-'))*dS(gamma_tag)
			fe -= (C_M/dt) * inner(fg, ve('+'))*dS(gamma_tag)		
		
		self.f =  [fi, fe]
					

	### class variables ###
		
	# physical parameters
	C_M     = 0.01
	sigma_i = 1.0
	sigma_e = 1.0
	
	# set scaling factor
	m_conversion_factor = 1

	# forcing factors
	source_i = Constant(0.0)
	source_e = Constant(0.0)

	# initial boundary potential 
	phi_e_init = Constant(0.0)

	# initial membrane potential 
	phi_M_init = Constant(-0.06774)							
	# phi_M_init = Expression('x[0]', degree=1)					
	# phi_M_init = Expression('0.5*sin(10*(x[0]*x[0] + x[1]*x[1]))', degree=4)					

	# order 
	fem_order = 1

	# BC
	dirichlet_bcs = False

