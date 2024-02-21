# Copyright © 2023 Pietro Benedusi
from dolfin       import *
from multiphenics import *
from EMIx.utils.MMS             import setup_MMS
from EMIx.utils.Mix_dim_problem import Mixed_dimensional_problem
import numpy as np
import time

# required by dS
parameters["ghost_mode"] = "shared_facet" 

# optimizaion flags
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3  -ffast-math'

parameters["form_compiler"]["quadrature_degree"] = 5 # TEST

#parameters["reorder_dofs_serial"] = False # TEST: False = paper ordering


class KNPEMI_problem(Mixed_dimensional_problem):
	

	def init(self):

		# set scaling factor
		self.m_conversion_factor =  1e-6

		# sources
		self.f_i = Expression('0', degree = 1, t = self.t)
		self.f_e = Expression('0', degree = 1, t = self.t)
		
		# for validation test
		if self.MMS_test: 
			self.setup_MMS_params() 		

	
	def setup_spaces(self):
		
		if MPI.comm_world.rank == 0: print('Creating spaces...') 

		# define elements
		P = FiniteElement('P', self.mesh.ufl_cell(), self.fem_order)

		# ion concentrations for each ion + potential
		element_list = [P]*(self.N_ions + 1)			

		self.V = FunctionSpace(self.mesh, MixedElement(element_list))
		
		self.W = BlockFunctionSpace([self.V, self.V], restrict=[self.interior, self.exterior])

		# create function for solution
		self.wh  = BlockFunction(self.W)

		# create function for solution at previous time step		
		self.u_p = BlockFunction(self.W)
		
		# rename for more readable output		
		self.u_p[0].rename('intra_ion', '')
		self.u_p[1].rename('extra_ion', '')

							
	def setup_boundary_conditions(self):

		# alias 
		Wi = self.W.sub(0)
		We = self.W.sub(1)

		# add Dirichlet boundary conditions on exterior boundary
		bci = []
		bce = []

		if self.dirichlet_bcs: 
			# bcs for concentrations
			# for idx, ion in enumerate(self.ion_list):

				#bc = DirichletBC(Wi.sub(idx), ion['ki_init'], self.boundaries, self.bound_tag)
				#bci.append(bc)

				#bc = DirichletBC(We.sub(idx), ion['ke_init'], self.boundaries, self.bound_tag)
				#bce.append(bc)			

			# for pontential 	
			bc = DirichletBC(Wi.sub(self.N_ions), self.phi_i_init, self.boundaries, self.bound_tag)
			bci.append(bc)		

			bc = DirichletBC(We.sub(self.N_ions), self.phi_e_init, self.boundaries, self.bound_tag)
			bce.append(bc)		

			self.bcs = BlockDirichletBC([bci, bce])
			
			# self.bcs = BlockDirichletBC([None, bce]) # old
		
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
		
			bc = DirichletBC(We.sub(self.N_ions), self.phi_e_init, point_bc, method="pointwise")
			bce.append(bc)		

			self.bcs = BlockDirichletBC([None, bce])

	
	def setup_variational_form(self):

		# sanity check
		if len(self.ionic_models) == 0 and MPI.comm_world.rank == 0:		
			print('\nERROR: call init_ionic_model() to provide ionic models!\n')
			exit()

		# aliases
		dt  = self.dt 
		F   = self.F
		psi = self.psi
		C_M = self.C_M
		t   = float(self.t)		
		
		if np.isclose(t,0): self.phi_M_prev = interpolate(self.phi_M_init, self.V.sub(self.N_ions).collapse())                  
				
		if MPI.comm_world.rank == 0: print('Setting up variational form...') 

		# define measures
		dx = Measure("dx")(subdomain_data=self.subdomains)
		dS = Measure("dS")(subdomain_data=self.boundaries)		

		dxi = dx(self.intra_tags)
		dxe = dx(self.extra_tag)			
		dS  = dS(self.gamma_tags) 


		# for the test various gamma faces get different tags
		if self.MMS_test:
						
			# for Omega_i = [0.25, 0.75] x [0.25, 0.75]
			gamma_subdomains = ('near(x[0], 0.25) and x[1] < 0.75 + 1e-12 and x[1] > 0.25 - 1e-12 ', \
								'near(x[0], 0.75) and x[1] < 0.75 + 1e-12 and x[1] > 0.25 - 1e-12 ', \
								'near(x[1], 0.25) and x[0] < 0.75 + 1e-12 and x[0] > 0.25 - 1e-12 ', \
								'near(x[1], 0.75) and x[0] < 0.75 + 1e-12 and x[0] > 0.25 - 1e-12 ', \
								'near(x[0], 0) or near(x[1], 0) or near(x[0], 1) or near(x[1], 1)')
			
			gamma = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1, 0)
			
			[subd.mark(gamma, i) for i, subd in enumerate(map(CompiledSubDomain, gamma_subdomains), 1)]  
			# redefine measure on gamma
			dS = Measure("dS")(subdomain_data=gamma)					
			dsOuter = dS(5)
			dS = dS((1,2,3,4))
			
		# init ionic models
		for model in self.ionic_models:			
			model._init()				
			
		# trial/test functions
		uu = BlockTrialFunction(self.W)
		vv = BlockTestFunction(self.W)

		(ui, ue) = block_split(uu)
		(vi, ve) = block_split(vv)

		# aliases		
		ui_p = self.u_p[0]
		ue_p = self.u_p[1]

		# rename for more readable output		
		ui_p.rename('intra_ion', '')
		ue_p.rename('extra_ion', '')

		# intracellular potential
		phi_i  = ui[self.N_ions]  # unknown
		vphi_i = vi[self.N_ions]  # test function

		# extracellular potential
		phi_e  = ue[self.N_ions]  # unknown
		vphi_e = ve[self.N_ions]  # test function
		
		# initialize
		alpha_i_sum = 0 # sum of fractions intracellular
		alpha_e_sum = 0 # sum of fractions extracellular		
		J_phi_i = 0     # total intracellular flux
		J_phi_e = 0     # total extracellular flux   

		# total channel current     
		I_ch = dict.fromkeys(self.gamma_tags, 0)		
		
		# Initialize parts of variational formulation
		for idx, ion in enumerate(self.ion_list):
			
			# get ion attributes
			z  = ion['z' ]; 
			Di = ion['Di']; 
			De = ion['De'];

			# set initial value of intra and extracellular ion concentrations
			if t == 0:
				assign(ui_p.sub(idx), interpolate(ion['ki_init'], self.W.sub(0).sub(idx).collapse()))
				assign(ue_p.sub(idx), interpolate(ion['ke_init'], self.W.sub(1).sub(idx).collapse()))			
		
			# add ion specific contribution to fraction alpha		
			alpha_i_sum += Di*z*z*ui_p.sub(idx)       
			alpha_e_sum += De*z*z*ue_p.sub(idx)
						
			# calculate and update Nernst potential for current ion
			ion['E'] = (psi/z)*ln(ue_p.sub(idx)/ui_p.sub(idx))		

			# init dictionary of ionic channel
			ion['I_ch'] = dict.fromkeys(self.gamma_tags) 			
						
			# loop ove ionic models
			for model in self.ionic_models:										

				# loop over ionic model tags
				for gamma_tag in model.tags:	
										
					ion['I_ch'][gamma_tag] = model._eval(idx)						

					# add contribution to total channel current							
					I_ch[gamma_tag] += ion['I_ch'][gamma_tag]										

		if t == 0:	
			# set phi_e and phi_i just for visualization
			assign(ui_p.sub(self.N_ions), interpolate(self.phi_i_init, self.W.sub(0).sub(self.N_ions).collapse()))			
			assign(ue_p.sub(self.N_ions), interpolate(self.phi_e_init, self.W.sub(1).sub(self.N_ions).collapse()))			

		# Initialize the variational form
		a00 = 0; a01 = 0; L0 = 0
		a10 = 0; a11 = 0; L1 = 0		
		
		# Setup ion specific part of variational formulation
		for idx, ion in enumerate(self.ion_list):
		 
			# get ion attributes
			z  = ion['z' ];
			Di = ion['Di'];
			De = ion['De'];
			I_ch_k = ion['I_ch']			

			# Set intracellular ion attributes
			ki  = ui[idx]          # unknown	
			vki = vi[idx]          # test function
			ki_prev = ui_p[idx]    # previous solution
			
			# Set extracellular ion attributes
			ke  = ue[idx]          # unknown			
			vke = ve[idx]          # test function
			ke_prev = ue_p[idx]    # previous solution
						
			# Set fraction of ion specific intra--and extracellular I_cap
			alpha_i = Di*z*z*ki_prev/alpha_i_sum
			alpha_e = De*z*z*ke_prev/alpha_e_sum			
			
			# linearised ion fluxes
			Ji = - Constant(Di)*grad(ki) - Constant(Di*z/psi)*ki_prev*grad(phi_i)
			Je = - Constant(De)*grad(ke) - Constant(De*z/psi)*ke_prev*grad(phi_e)		
			
			# some useful constants
			C_i = C_M*alpha_i('-')/(F*z)
			C_e = C_M*alpha_e('-')/(F*z)

			# weak form - equation for k_i
			a00 += ki*vki*dxi - dt * inner(Ji, grad(vki))*dxi + C_i * inner(phi_i('-'),vki('-'))*dS
			a01 += - C_i * inner(phi_e('-'),vki('-'))*dS
			L0  += ki_prev*vki*dxi

			# weak form - equation for k_e
			a11 += ke*vke*dxe - dt * inner(Je, grad(vke))*dxe + C_e * inner(phi_e('-'),vke('-'))*dS
			a10 += - C_e * inner(phi_i('-'),vke('-'))*dS
			L1  += ke_prev*vke*dxe 

			# ionic channels (can be in dS subset)
			for gamma_tag in self.gamma_tags:						

				L0 -= (dt*I_ch_k[gamma_tag] - alpha_i('-')*C_M*self.phi_M_prev)/(F*z)*vki('-')*dS(gamma_tag)
				L1 += (dt*I_ch_k[gamma_tag] - alpha_e('-')*C_M*self.phi_M_prev)/(F*z)*vke('-')*dS(gamma_tag)
			
			# add contribution to total current flux
			J_phi_i += z*Ji
			J_phi_e += z*Je

			# source terms
			L0 += inner(ion['f_i']*self.f_i,vki)*dxi									
			L1 += inner(ion['f_e']*self.f_e,vke)*dxe									
							
			if self.MMS_test:
				# define outward normal on exterior boundary (partial Omega)
				self.n_outer = FacetNormal(self.mesh)				
				
				L0 += dt * inner(ion['f_k_i'], vki)*dxi # eq for k_i			
				L1 += dt * inner(ion['f_k_e'], vke)*dxe # eq for k_e

				# enforcing correction for I_m 
				for i, JM in enumerate(ion['f_I_M'], 1):						
					L0 += dt/(F*z) * alpha_i('-')*inner(JM, vki('-'))*dS(i)
					L1 -= dt/(F*z) * alpha_e('-')*inner(JM, vke('-'))*dS(i)

				# enforcing correction for I_m = -Fsum(zJ_e*ne) + sum(gM_k) (Assueme gM_k = gM/N_ions)
				L1 -= dt/(F*z) * sum(alpha_e('-')*inner(gM, vke('-'))*dS(i) for i, gM in enumerate(ion['f_g_M'], 1))								
								
				# exterior boundary terms (zero in "physical" problem)
				L1 -=  dt*inner(dot(ion['J_k_e'], self.n_outer('-')), vke('-')   )*dsOuter # eq for k_e
				L1 += F*z*inner(dot(ion['J_k_e'], self.n_outer('-')), vphi_e('-'))*dsOuter # eq for phi_e		

		# weak form - equation for phi_i		
		a00 -= inner(J_phi_i, grad(vphi_i))*dxi - (C_M/(F*dt)) * inner(phi_i('-'),vphi_i('-'))*dS
		a01 -= (C_M/(F*dt)) * inner(phi_e('-'),vphi_i('-'))*dS		
		
		# weak form - equation for phi_e
		a11 -= inner(J_phi_e, grad(vphi_e))*dxe - (C_M/(F*dt)) * inner(phi_e('-'),vphi_e('-'))*dS
		a10 -= (C_M/(F*dt)) * inner(phi_i('-'),vphi_e('-'))*dS		

		for gamma_tag in self.gamma_tags:				

			L0  -= (1/F)*(I_ch[gamma_tag] - C_M*self.phi_M_prev/dt)*vphi_i('-')*dS(gamma_tag)
			L1  += (1/F)*(I_ch[gamma_tag] - C_M*self.phi_M_prev/dt)*vphi_e('-')*dS(gamma_tag)

		if self.MMS_test:
			# phi source terms					
			L0 -= inner(ion['f_phi_i'], vphi_i)*dxi # equation for phi_i
			L1 -= inner(ion['f_phi_e'], vphi_e)*dxe # equation for phi_e			

			# enforcing correction for I_m = dphi/dt + I_ch -JM
			for i, JM in enumerate(ion['f_I_M'], 1):
				L0 += inner(JM, vphi_i('-'))*dS(i) 
				L1 -= inner(JM, vphi_e('-'))*dS(i) 
			
			# enforcing correction for I_m = -Fsum(zJ_e*ne) + gM
			L1 -= sum(inner(gM, vphi_e('-'))*dS(i) for i, gM in enumerate(ion['f_g_M'], 1))			
					
		# gather weak form in matrix structure
		self.a = [[a00, a01],
				  [a10, a11]]

		self.L = [L0, L1]
	

	def setup_preconditioner(self, use_block_jacobi):

		# aliases
		dt  = self.dt 
		F   = self.F
		psi = self.psi
		C_M = self.C_M		
		
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

		# aliases		
		ui_p = self.u_p[0]
		ue_p = self.u_p[1]

		# intracellular potential
		phi_i  = ui[self.N_ions]  # unknown
		vphi_i = vi[self.N_ions]  # test function

		# extracellular potential
		phi_e  = ue[self.N_ions]  # unknown
		vphi_e = ve[self.N_ions]  # test function
		
		# initialize
		J_phi_i = 0     # total intracellular flux
		J_phi_e = 0     # total extracellular flux        
		
		# Initialize the variational form
		p00 = 0; p01 = 0; 
		p10 = 0; p11 = 0; 
		
		# Setup ion specific part of variational formulation
		for idx, ion in enumerate(self.ion_list):
		 
			# get ion attributes
			z  = ion['z' ];
			Di = ion['Di'];
			De = ion['De'];						

			# Set intracellular ion attributes
			ki  = ui[idx]          # unknown	
			vki = vi[idx]          # test function
			ki_prev = ui_p[idx]    # previous solution
			
			# Set extracellular ion attributes
			ke  = ue[idx]          # unknown			
			vke = ve[idx]          # test function
			ke_prev = ue_p[idx]    # previous solution			
						
			# linearised ion fluxes			
			if use_block_jacobi:
				
				Ji = - Constant(Di*z/psi)*ki_prev*grad(phi_i)
				Je = - Constant(De*z/psi)*ke_prev*grad(phi_e)																						
			
			else:

				Ji = - Constant(Di)*grad(ki) - Constant(Di*z/psi)*ki_prev*grad(phi_i)
				Je = - Constant(De)*grad(ke) - Constant(De*z/psi)*ke_prev*grad(phi_e)																						
										
			# weak form - equation for k_i
			p00 += ki*vki*dxi + dt * inner(Constant(Di)*grad(ki), grad(vki))*dxi 			

			# weak form - equation for k_e
			p11 += ke*vke*dxe + dt * inner(Constant(De)*grad(ke), grad(vke))*dxe 			
			
			# add contribution to total current flux
			J_phi_i += z*Ji
			J_phi_e += z*Je

		# weak form - equation for phi_i		
		p00 -= inner(J_phi_i, grad(vphi_i))*dxi - (C_M/(F*dt)) * inner(phi_i('-'),vphi_i('-'))*dS		
		
		# weak form - equation for phi_e
		p11 -= inner(J_phi_e, grad(vphi_e))*dxe - (C_M/(F*dt)) * inner(phi_e('-'),vphi_e('-'))*dS		

		# gather weak form in matrix structure
		self.P = [[p00, 0],
				  [0, p11]]
	
	
	def setup_MMS_params(self):

		self.dirichlet_bcs       = True		
		self.m_conversion_factor = 1
		
		self.C_M = 1
		self.F   = 1
		self.R   = 1
		self.T   = 1   
		self.psi = 1           
		
		self.M = setup_MMS()

		src_terms, exact_sols, init_conds, bndry_terms, subdomains_MMS = self.M.get_MMS_terms_KNPEMI(self.t)

		# initial values
		self.phi_M_init = init_conds['phi_M']   # membrane potential (V)
		self.phi_i_init = exact_sols['phi_i_e'] # internal potential (V) just for visualization
		self.phi_e_init = exact_sols['phi_e_e'] # external potential (V) just for visualization	

		# create ions
		self.Na = {'Di':1.0, 'De':1.0, 'z':1.0,
			  'ki_init':init_conds['Na_i'],
			  'ke_init':init_conds['Na_e'],					   
			  'f_k_i':src_terms['f_Na_i'],
			  'f_k_e':src_terms['f_Na_e'],
			  'J_k_e':bndry_terms['J_Na_e'],
			  'phi_i_e':exact_sols['phi_i_e'],
			  'phi_e_e':exact_sols['phi_e_e'],
			  'f_phi_i':src_terms['f_phi_i'],
			  'f_phi_e':src_terms['f_phi_e'],
			  'f_g_M':src_terms['f_g_M'],
			  'f_I_M':src_terms['f_I_M'],
			  'name':'Na'}

		self.K = {'Di':1.0, 'De':1.0, 'z':1.0,
			 'ki_init':init_conds['K_i'],
			 'ke_init':init_conds['K_e'],			 			 
			 'f_k_i':src_terms['f_K_i'],
			 'f_k_e':src_terms['f_K_e'],
			 'J_k_e':bndry_terms['J_K_e'],
			 'phi_i_e':exact_sols['phi_i_e'],
			 'phi_e_e':exact_sols['phi_e_e'],
			 'f_phi_i':src_terms['f_phi_i'],
			 'f_phi_e':src_terms['f_phi_e'],
			 'f_g_M':src_terms['f_g_M'],
			 'f_I_M':src_terms['f_I_M'],
			  'name':'K'}

		self.Cl = {'Di':1.0, 'De':1.0, 'z':-1.0,
			  'ki_init':init_conds['Cl_i'],
			  'ke_init':init_conds['Cl_e'],		
			  'f_k_i':src_terms['f_Cl_i'],
			  'f_k_e':src_terms['f_Cl_e'],
			  'J_k_e':bndry_terms['J_Cl_e'],
			  'phi_i_e':exact_sols['phi_i_e'],
			  'phi_e_e':exact_sols['phi_e_e'],
			  'f_phi_i':src_terms['f_phi_i'],
			  'f_phi_e':src_terms['f_phi_e'],
			  'f_g_M':src_terms['f_g_M'],
			  'f_I_M':src_terms['f_I_M'],
			  'name':'Cl'}

		# create ion list
		self.ion_list = [self.Na, self.K, self.Cl]		


	def print_conservation(self):
		
		# define measures
		dx = Measure("dx")(subdomain_data=self.subdomains)
		dxi = dx(self.intra_tags)
		dxe = dx(self.extra_tag)

		Na_i  = self.wh[0].sub(0)
		K_i   = self.wh[0].sub(1)
		Cl_i  = self.wh[0].sub(2)		

		Na_e  = self.wh[1].sub(0)
		K_e   = self.wh[1].sub(1)
		Cl_e  = self.wh[1].sub(2)		

		Na_tot = assemble(Na_i*dxi) + assemble(Na_e*dxe) 
		K_tot  = assemble(K_i *dxi) + assemble(K_e*dxe) 
		Cl_tot = assemble(Cl_i*dxi) + assemble(Cl_e*dxe) 

		print("Na tot:", Na_tot)
		print("K tot:",  K_tot)
		print("Cl tot:", Cl_tot)		


	def print_errors(self):

		# get exact solutions
		src_terms, exact_sols, init_conds, bndry_terms, subdomains_MMS = self.M.get_MMS_terms_KNPEMI(self.t)

		# define measures
		dx = Measure("dx")(subdomain_data=self.subdomains)
		dxi = dx(self.intra_tags)
		dxe = dx(self.extra_tag)

		Na_i  = self.wh[0].sub(0)
		K_i   = self.wh[0].sub(1)
		Cl_i  = self.wh[0].sub(2)
		phi_i = self.wh[0].sub(3)

		Na_e  = self.wh[1].sub(0)
		K_e   = self.wh[1].sub(1)
		Cl_e  = self.wh[1].sub(2)
		phi_e = self.wh[1].sub(3)

		err_Na_i  = inner(Na_i  - exact_sols['Na_i_e'],  Na_i  - exact_sols['Na_i_e']) *dxi
		err_K_i   = inner(K_i   - exact_sols['K_i_e'],   K_i   - exact_sols['K_i_e'])  *dxi
		err_Cl_i  = inner(Cl_i  - exact_sols['Cl_i_e'],  Cl_i  - exact_sols['Cl_i_e']) *dxi
		err_phi_i = inner(phi_i - exact_sols['phi_i_e'], phi_i - exact_sols['phi_i_e'])*dxi
		
		err_Na_e  = inner(Na_e  - exact_sols['Na_e_e'],  Na_e  - exact_sols['Na_e_e']) *dxe
		err_K_e   = inner(K_e   - exact_sols['K_e_e'],   K_e   - exact_sols['K_e_e'])  *dxe
		err_Cl_e  = inner(Cl_e  - exact_sols['Cl_e_e'],  Cl_e  - exact_sols['Cl_e_e']) *dxe
		err_phi_e = inner(phi_e - exact_sols['phi_e_e'], phi_e - exact_sols['phi_e_e'])*dxe
		
		L2_err_Na_i  = sqrt(assemble(err_Na_i))
		L2_err_K_i   = sqrt(assemble(err_K_i))
		L2_err_Cl_i  = sqrt(assemble(err_Cl_i))
		L2_err_phi_i = sqrt(assemble(err_phi_i))

		L2_err_Na_e  = sqrt(assemble(err_Na_e))
		L2_err_K_e   = sqrt(assemble(err_K_e))
		L2_err_Cl_e  = sqrt(assemble(err_Cl_e))
		L2_err_phi_e = sqrt(assemble(err_phi_e))

		print("~~~~~~~~~~~~~~ Errors ~~~~~~~~~~~~~~")
		print('L2 Na_i  error:', L2_err_Na_i)
		print('L2 Na_e  error:', L2_err_Na_e)
		print('L2 K_i   error:', L2_err_K_i)
		print('L2 K_e   error:', L2_err_K_e)
		print('L2 Cl_i  error:', L2_err_Cl_i)
		print('L2 Cl_e  error:', L2_err_Cl_e)
		print('L2 phi_i error:', L2_err_phi_i)
		print('L2 phi_e error:', L2_err_phi_e)


	### class variables ###	
	
	# physical parameters
	C_M = 0.02                       # capacitance (F)
	T   = 300                        # temperature (K)
	F   = 96485                      # Faraday's constant (C/mol)
	R   = 8.314                      # Gas constant (J/(K*mol))
	psi = R*T/F                      # recurring variable (psi = 0.0259, 1/psi = 38.7)
	g_Na_bar  = 1200                 # Na max conductivity (S/m**2)
	g_K_bar   = 360                  # K max conductivity (S/m**2)    
	g_Na_leak = Constant(2.0*0.5)    # Na leak conductivity (S/m**2)
	g_K_leak  = Constant(8.0*0.5)    # K leak conductivity (S/m**2)
	g_Cl_leak = Constant(0.0)        # Cl leak conductivity (S/m**2)
	a_syn     = 0.002                # synaptic time constant (s)
	g_syn_bar = 40                   # synaptic conductivity (S/m**2)
	D_Na = Constant(1.33e-9)         # diffusion coefficients Na (m/s)
	D_K  = Constant(1.96e-9)         # diffusion coefficients K (m/s)
	D_Cl = Constant(2.03e-9)         # diffusion coefficients Cl (m/s)
	V_rest  = -0.065                 # resting membrane potential
	# E_Na    = 54.8e-3                # reversal potential Na (V)
	# E_K     = -88.98e-3              # reversal potential K (V)
	
	# potassium buffering params
	rho_pump = 1.115e-6			     # maximum pump rate (mol/m**2 s)
	P_Nai = 10                       # [Na+]i threshold for Na+/K+ pump (mol/m^3)
	P_Ke  = 1.5                      # [K+]e  threshold for Na+/K+ pump (mol/m^3)
	k_dec = 2.9e-8				     # Decay factor for [K+]e (m/s)

	# initial conditions
	phi_e_init = Constant(0)         # external potential (V)
	phi_i_init = Constant(-0.06774)  # internal potential (V) just for visualization
	phi_M_init = Constant(-0.06774)  # membrane potential (V)	
	Na_i_init  = Constant(12)        # intracellular Na concentration (mol/m^3)
	Na_e_init  = Constant(100)       # extracellular Na concentration (mol/m^3)
	K_i_init   = Constant(125)       # intracellular K  concentration (mol/m^3)
	K_e_init   = Constant(4)         # extracellular K  concentration (mol/m^3)
	Cl_i_init  = Constant(137)       # intracellular Cl concentration (mol/m^3)
	Cl_e_init  = Constant(104)       # extracellular Cl concentration (mol/m^3)

	# initial gating
	n_init = Constant(0.27622914792) # gating variable n
	m_init = Constant(0.03791834627) # gating variable m
	h_init = Constant(0.68848921811) # gating variable h

	# sources
	Na_e_f = Constant(0.0)
	Na_i_f = Constant(0.0)
	K_e_f  = Constant(0.0)
	K_i_f  = Constant(0.0)
	Cl_e_f = Constant(0.0)
	Cl_i_f = Constant(0.0)

	# create ions (Na conductivity is set below for each model)
	Na = {'g_leak':g_Na_leak,'Di':D_Na,'De':D_Na,'ki_init':Na_i_init,'ke_init':Na_e_init,'z':1.0, 'name':'Na','f_e': Na_e_f,'f_i':Na_i_f,'rho_p': 3*rho_pump}
	K  = {'g_leak':g_K_leak, 'Di':D_K, 'De':D_K, 'ki_init':K_i_init, 'ke_init':K_e_init, 'z':1.0, 'name':'K' ,'f_e': K_e_f, 'f_i':K_i_f, 'rho_p':-2*rho_pump}
	Cl = {'g_leak':g_Cl_leak,'Di':D_Cl,'De':D_Cl,'ki_init':Cl_i_init,'ke_init':Cl_e_init,'z':-1.0,'name':'Cl','f_e': Cl_e_f,'f_i':Cl_i_f,'rho_p':0.0}
	
	# create ion list
	ion_list = [Na, K, Cl]
	N_ions = len(ion_list) 

	# order 
	fem_order = 1
	
	# test flag
	MMS_test = False	

	# BC (only on phi)
	dirichlet_bcs = False








		