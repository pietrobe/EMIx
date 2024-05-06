# Copyright © 2023 Pietro Benedusi
from abc    import ABC, abstractmethod
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import time

# zero stimulus (default)
def g_syn_none(g_syn_bar, a_syn, t):		
	return Constant(0.0)


class Ionic_model(ABC):

	# constructor
	def __init__(self, EMI_problem, tags=None):

		self.problem = EMI_problem	
		self.tags = tags

		# if tags are not specified we use all the intra tags
		if self.tags == None: self.tags = self.problem.gamma_tags

		# trasform int in tuple if needed
		if isinstance(self.tags, int): self.tags = (self.tags,)
	
			
	@abstractmethod
	def _eval(self):
		# Abstract method that must be implemented by concrete subclasses.
		pass


# I_ch = phi_M
class Passive_model(Ionic_model):
	
	def __str__(self):
		return f'Passive'
		
	def _eval(self):			

		I_ch = self.problem.phi_M	
		return I_ch


# Hodgkin–Huxley + stimuls
class HH_model(Ionic_model):	

	# HH params
	# initial gating	
	n_init = Constant(0.27622914792) # gating variable n
	m_init = Constant(0.03791834627) # gating variable m
	h_init = Constant(0.68848921811) # gating variable h

	# conductivities
	g_Na_bar  = 1200                 # Na max conductivity (S/m**2)
	g_K_bar   = 360                  # K max conductivity (S/m**2)    
	g_Na_leak = Constant(2.0*0.5)    # Na leak conductivity (S/m**2)
	g_K_leak  = Constant(8.0*0.5)    # K leak conductivity (S/m**2)
	g_Cl_leak = Constant(0.0)        # Cl leak conductivity (S/m**2)		
	V_rest    = -0.065               # resting membrane potential
	E_Na      = 54.8e-3              # reversal potential Na (V)
	E_K       = -88.98e-3            # reversal potential K (V)
	E_Cl      = 0  		             # reversal potential 0 (V)
	
	# numerics
	use_Rush_Lar   = True
	time_steps_ODE = 26

	# save gating in PNG	
	save_png_file = True		
		

	def __init__(self, EMI_problem, tags=None, stim_fun=g_syn_none):

		super().__init__(EMI_problem, tags)

		self.g_Na_stim = stim_fun


	def __str__(self):
		return f'Hodgkin–Huxley'
	
	
	def _eval(self):	

		# aliases		
		p = self.problem						

		# update gating variables
		if float(p.t) == 0:
					
			self.n = interpolate(self.n_init, p.V)			
			self.m = interpolate(self.m_init, p.V)
			self.h = interpolate(self.h_init, p.V)	

			# output
			if self.save_png_file: self.init_png()

		else:
			self.update_gating_variables()	

			# output
			if self.save_png_file: self.save_png()					
		
		# conductivities
		g_Na = self.g_Na_leak + self.g_Na_bar*self.m**3*self.h
		g_K  = self.g_K_leak  + self.g_K_bar*self.n**4				
		g_Cl = self.g_Cl_leak

		# stimulus
		g_Na += self.g_Na_stim(float(p.t)) 

		# ionic currents
		I_ch_Na = g_Na * (p.phi_M - self.E_Na)
		I_ch_K  = g_K  * (p.phi_M - self.E_K)
		I_ch_Cl = g_Cl * (p.phi_M - self.E_Cl)		
		
		# total current
		I_ch = I_ch_Na + I_ch_K + I_ch_Cl
		
		return I_ch


	def update_gating_variables(self):			
		
		tic = time.perf_counter()	

		# aliases			
		n = self.n
		m = self.m
		h = self.h
		phi_M = self.problem.phi_M
		dt_ode = float(self.problem.dt)/self.time_steps_ODE	
		
		V_M = 1000*(phi_M.vector()[:] - self.V_rest) # convert phi_M to mV						
		
		alpha_n = 0.01e3*(10.-V_M)/(np.exp((10.-V_M)/10.) - 1.)
		beta_n  = 0.125e3*np.exp(-V_M/80.)
		alpha_m = 0.1e3*(25. - V_M)/(np.exp((25. - V_M)/10.) - 1)
		beta_m  = 4.e3*np.exp(-V_M/18.)
		alpha_h = 0.07e3*np.exp(-V_M/20.)
		beta_h  = 1.e3/(np.exp((30.-V_M)/10.) + 1)

		if self.use_Rush_Lar:
			
			tau_y_n = 1.0/(alpha_n + beta_n)
			tau_y_m = 1.0/(alpha_m + beta_m)
			tau_y_h = 1.0/(alpha_h + beta_h)

			y_inf_n = alpha_n * tau_y_n
			y_inf_m = alpha_m * tau_y_m
			y_inf_h = alpha_h * tau_y_h

			y_exp_n =  np.exp(-dt_ode/tau_y_n);
			y_exp_m =  np.exp(-dt_ode/tau_y_m);
			y_exp_h =  np.exp(-dt_ode/tau_y_h);
			
		else:

			alpha_n *= dt_ode
			beta_n  *= dt_ode
			alpha_m *= dt_ode
			beta_m  *= dt_ode
			alpha_h *= dt_ode
			beta_h  *= dt_ode
		
		for i in range(self.time_steps_ODE): 

			if self.use_Rush_Lar:

				n.vector()[:] = y_inf_n + (n.vector()[:] - y_inf_n) * y_exp_n
				m.vector()[:] = y_inf_m + (m.vector()[:] - y_inf_m) * y_exp_m
				h.vector()[:] = y_inf_h + (h.vector()[:] - y_inf_h) * y_exp_h
				
			else:

				n.vector()[:] += alpha_n * (1 - n.vector()[:]) - beta_n * n.vector()[:]
				m.vector()[:] += alpha_m * (1 - m.vector()[:]) - beta_m * m.vector()[:]
				h.vector()[:] += alpha_h * (1 - h.vector()[:]) - beta_h * h.vector()[:]	

		toc = time.perf_counter()
		if MPI.comm_world.rank == 0: print(f"ODE step in {toc - tic:0.4f} seconds")   	


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
		
		local_n = self.n.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
		local_m = self.m.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
		local_h = self.h.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
		
		self.n_t = []
		self.m_t = []
		self.h_t = []
		
		self.n_t.append(local_n[self.point_to_plot]) 
		self.m_t.append(local_m[self.point_to_plot]) 
		self.h_t.append(local_h[self.point_to_plot]) 
		
		self.out_gate_string = 'output/gating.png'
			

	def save_png(self):
		
		p = self.problem

		# prepare data (needed for parallel)
		dmap = p.V.dofmap()		
		imap = dmap.index_map()
		num_dofs_local = imap.size(IndexMap.MapSize.ALL) * imap.block_size()

		local_n = self.n.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
		local_m = self.m.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
		local_h = self.h.vector().get_local(np.arange(num_dofs_local, dtype=np.int32))
					
		self.n_t.append(local_n[self.point_to_plot]) 
		self.m_t.append(local_m[self.point_to_plot]) 
		self.h_t.append(local_h[self.point_to_plot]) 


	def plot_png(self):
		
		# aliases
		dt = float(self.problem.dt)
		
		time_steps = len(self.n_t)

		plt.figure(1)
		plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps), self.n_t, label='n')
		plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps), self.m_t, label='m')
		plt.plot(np.linspace(0, 1000*time_steps*dt, time_steps), self.h_t, label='h')
		plt.legend()
		plt.xlabel('time (ms)')
		plt.savefig(self.out_gate_string)