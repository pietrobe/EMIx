from abc    import ABC, abstractmethod
from dolfin import *
import numpy as np
import time

# stimulus
def g_syn(g_syn_bar, a_syn, t):
		
	g = Expression('g_syn_bar*exp(-fmod(t,0.01)/a_syn)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)			
	# g = Expression('g_syn_bar', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)			
	# g = Expression('g_syn_bar*exp(-fmod(t,0.01)/a_syn)*(x[2] < 5e-4)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)		
	# g = Expression('g_syn_bar*(x[0] < 0.3) * (x[1] < 0.3)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)		

	return g


# Kir-function used in ionic pump
def f_Kir(K_e_init, K_e, EK_init, Dphi, phi_m):

	A = 1 + exp(18.4/42.4)
	B = 1 + exp(-(0.1186 + EK_init)/0.0441)
	C = 1 + exp((Dphi + 0.0185)/0.0425)
	D = 1 + exp(-(0.1186 + phi_m)/0.0441)

	f = sqrt(K_e/K_e_init) * A*B/(C*D)

	return f

#################################

class Ionic_model(ABC):

	# constructor
	def __init__(self, KNPEMI_problem, tags=None):

		self.problem = KNPEMI_problem	
		self.tags = tags

		# if tags are not specified we use all the intra tags
		if self.tags == None:
			self.tags = self.problem.gamma_tags

		# trasform int in tuple if needed
		if isinstance(self.tags, int): self.tags = (self.tags,)
	

	@abstractmethod
	def _init(self):
		# Abstract method that must be implemented by concrete subclasses.
		# Init ion-independent quantities
		pass
			
	@abstractmethod
	def _eval(self, ion_idx):
		# Abstract method that must be implemented by concrete subclasses.
		pass


# I_ch = phi_M
class Null_model(Ionic_model):

	def _init(self):		
		pass

	def __str__(self):
		return f'Zero'
		
	def _eval(self, ion_idx):			
		return 0


# I_ch = phi_M
class Passive_model(Ionic_model):

	def _init(self):		
		pass

	def __str__(self):
		return f'Passive'
		
	def _eval(self, ion_idx):	
		I_ch = self.problem.phi_M_prev	
		return I_ch


# I_ch = g*(phi_M - E) + stimuls
class Passive_Nerst_model(Ionic_model):

	def __init__(self, KNPEMI_problem, tags=None, stimuls=True):

		super().__init__(KNPEMI_problem, tags)

		self.stimuls = stimuls

	def __str__(self):
		return f'Passive'
	
	def _init(self):			
		pass

	def _eval(self, ion_idx):	
		
		# aliases	
		p     = self.problem			
		ion   = p.ion_list[ion_idx]
		phi_M = p.phi_M_prev

		# leak currents
		ion['g_k'] = ion['g_leak']

		# stimulus
		if ion['name'] == 'Na' and self.stimuls:
			ion['g_k'] += g_syn(p.g_syn_bar, p.a_syn, float(p.t)) 

		I_ch = ion['g_k']*(phi_M - ion['E'])
		
		return I_ch


# Ionic K pump
class Passive_K_pump_model(Ionic_model):

	# -k_dec * ([K]e − [K]e_0) both for K and Na
	use_decay_currents = False			

	def __str__(self):
		if self.use_decay_currents:
			return f'Passive with K pump and decay currents'
		else:
			return f'Passive with K pump'


	def _init(self):

		# aliases		
		p = self.problem

		ui_p  = self.problem.u_p[0]
		ue_p  = self.problem.u_p[1]
		P_Nai = self.problem.P_Nai
		P_Ke  = self.problem.P_Ke

		self.pump_coeff = ui_p.sub(0)**1.5/(ui_p.sub(0)**1.5 + P_Nai**1.5) * (ue_p.sub(1)/(ue_p.sub(1) + P_Ke))			
		

	def _eval(self, ion_idx):

		# aliases		
		p = self.problem
		
		phi_M = p.phi_M_prev		
		ue_p  = p.u_p[1]
		F     = p.F
		ion   = p.ion_list[ion_idx]
		z     = ion['z' ]
		K_e_init = p.K_e_init

		# leak currents
		ion['g_k'] = ion['g_leak']
			
		# f kir coeff	
		if ion['name'] == 'K':

			EK_init = (p.psi/z)*ln(K_e_init/p.K_i_init)			
			Dphi  = phi_M - ion['E']
			f_kir = f_Kir(K_e_init, ue_p.sub(ion_idx), EK_init, Dphi, phi_M)

		else:
			f_kir = 1

		I_ch = f_kir*ion['g_k']*(phi_M - ion['E']) + F*z*ion['rho_p']*self.pump_coeff

		if self.use_decay_currents:
			if ion['name'] == 'K' or ion['name'] == 'Na':
				I_ch -= F*z*p.k_dec*(ue_p.sub(1) - K_e_init)  
		
		return I_ch


# Hodgkin–Huxley + stimuls
class HH_model(Ionic_model):
	
	def __init__(self, KNPEMI_problem, tags=None, stimuls=True, use_Rush_Lar = True, time_steps_ODE = 25):

		super().__init__(KNPEMI_problem, tags)

		self.stimuls = stimuls
		self.use_Rush_Lar = use_Rush_Lar
		self.time_steps_ODE = time_steps_ODE
 	

	def __str__(self):
		return f'Hodgkin–Huxley'

	def _init(self):	
		
		# alias
		p = self.problem		
		
		# update gating variables
		if float(p.t) == 0:
					
				p.n = interpolate(p.n_init, p.V.sub(p.N_ions).collapse())			
				p.m = interpolate(p.m_init, p.V.sub(p.N_ions).collapse())
				p.h = interpolate(p.h_init, p.V.sub(p.N_ions).collapse())	

		else:
			self.update_gating_variables()			
		
	
	def _eval(self, ion_idx):	

		# aliases		
		p     = self.problem		
		ion   = p.ion_list[ion_idx]
		phi_M = p.phi_M_prev

		# leak currents
		ion['g_k'] = ion['g_leak']

		# stimulus and gating
		if ion['name'] == 'Na':
			if self.stimuls: ion['g_k'] += g_syn(p.g_syn_bar, p.a_syn, float(p.t)) 
			ion['g_k'] += p.g_Na_bar*p.m**3*p.h

		elif ion['name'] == 'K':
			ion['g_k'] += p.g_K_bar*p.n**4				
		
		I_ch = ion['g_k']*(phi_M - ion['E'])
		
		return I_ch


	def update_gating_variables(self):		
		
		tic = time.perf_counter()	

		# aliases			
		n = self.problem.n
		m = self.problem.m
		h = self.problem.h
		phi_M_prev = self.problem.phi_M_prev
		dt_ode = float(self.problem.dt)/self.time_steps_ODE	
		
		V_M = 1000*(phi_M_prev.vector()[:] - self.problem.V_rest) # convert phi_M to mV				

		# # correction to prevent overflow for indeces not in gamma (TODO PARALLEL?)
		# V_M[abs(V_M)>200] = 0

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
			
	









