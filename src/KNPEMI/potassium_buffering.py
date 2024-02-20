import KNPEMI_problem as KNPEMI_p
import KNPEMI_solver  as KNPEMI_s
from dolfin import *

# 100 ms injection in 1-2 micro m zone
# peak increase at 10 mM
# tot time 0.1 s?

# TODO: potential is not going to 0 after stimulus?? Yes with no K pump enabled

if __name__=='__main__':
	
	# global time step (s)
	dt = 0.0005
	time_steps = 2000	

	# # single astocyte
	# input_files = [ "data/astro_in_ball/mesh0.xdmf", "", "data/astro_in_ball/facets0.xdmf", \
	# 				"data/astro_in_ball/interior_restriction0.rtc.xdmf",\
	# 				"data/astro_in_ball/exterior_restriction0.rtc.xdmf", ""]		
	
	# cylinder
	input_files = [ "data/cylinder/mesh0.xdmf", "", "data/cylinder/facets0.xdmf", \
					"data/cylinder/interior_restriction0.rtc.xdmf",\
					"data/cylinder/exterior_restriction0.rtc.xdmf", ""]		
	
	# tags for intra, extra, bound, gamma
	tags = [2, 1, 4, 3] 
	
	# create KNP-EMI problem and solver
	problem = KNPEMI_p.Problem(input_files, tags, dt)

	# ionic model
	problem.HH_model    = False
	problem.enable_pump = True

	# update params
	problem.C_M = 0.01                       
	problem.T   = 298                        
	problem.F   = 96500     
	problem.R   = 8.314      
	problem.psi = problem.R*problem.T/problem.F 
	
	problem.g_Na_leak = 1          
	problem.g_K_leak  = 16.96  
	problem.g_Cl_leak = 0.5             	
		
	# initial conditions
	problem.Na['ki_init'] = Constant(15.189)      
	problem.Na['ke_init'] = Constant(144.662)     
	problem.K['ki_init']  = Constant(99.959)        
	problem.K['ke_init']  = Constant(3.082)     
	problem.Cl['ki_init'] = Constant(5.145)    
	problem.Cl['ke_init'] = Constant(133.71)    

	problem.phi_e_init = Constant(0)       
	problem.phi_i_init = Constant(-0.0859) 
	problem.phi_M_init = Constant(-0.0859)	

	problem.update()
	
	# solve
	solver  = KNPEMI_s.Solver(problem, time_steps)

