from EMIx.EMI.EMI_problem import EMI_problem
from EMIx.EMI.EMI_solver  import EMI_solver  
from dolfin import *
import time
from sys   import argv


#  Na stimulus definition
def g_Na_stim(g_syn_bar, a_syn, t):	

	# Single			
	g = Expression('g_syn_bar*exp(-fmod(t,0.02)/a_syn)*(x[2] < -4.4e-4)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)	

	# # Multi
	# g = Expression('g_syn_bar*exp(-fmod(t,0.02)/a_syn)*(x[2] < -4.8e-3)*(x[0]*x[0] + x[1]*x[1] >= pow(0.35, 2))', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)	
	
	return g

if __name__=='__main__':
		
	# global time step (s)	
	dt = 0.00005
	time_steps = 200

	# # grid size
	# N = int(argv[1])
	
	# # square	
	# input_path  = "../../../data/square/"
	# input_files = {'mesh_file':             input_path + "square"                  + str(N) + ".xml", \
	#  			   'subdomais_file': 		input_path + "square_physical_region"  + str(N) + ".xml", \
	#  			   'facets_file':           input_path + "square_facet_region"     + str(N) + ".xml", \
	#  			   'intra_restriction_dir': input_path + "square_restriction_om_i" + str(N) + ".rtc.xml", \
	#  			   'extra_restriction_dir': input_path + "square_restriction_om_e" + str(N) + ".rtc.xml"}		
	
	# tags = {'intra': 1 , 'extra': 2, 'boundary': 1, 'membrane': 2}	

	# # CL 	
	# input_path  = "data/CL/"
	# input_files = {'mesh_file':             input_path + "mesh0.xdmf",\
	# 			   'facets_file':           input_path + "facets0.xdmf",\
	# 			   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf",\
	# 			   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}	
		   	
	# tags = {'intra': 2, 'extra': 1, 'boundary': 4, 'membrane': 2}	

	# CL myelin	
	input_path  = "data/CLmyel/"
	input_files = {'mesh_file':             input_path + "mesh0.xdmf",\
				   'facets_file':           input_path + "facets0.xdmf",\
				   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf",\
				   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}	
		   	
	tags = {'intra': 3, 'extra': 1, 'boundary': 4, 'membrane': 2}	

	# # CL myelin	
	# input_path  = "data/CLmyel_retract/"
	# input_files = {'mesh_file':             input_path + "mesh0.xdmf",\
	# 			   'facets_file':           input_path + "facets0.xdmf",\
	# 			   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf",\
	# 			   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}	
		   	
	# tags = {'intra': 3, 'extra': 1, 'boundary': 4, 'membrane': 2}	

	# # CL9 	
	# input_path  = "data/CL9/"
	# input_files = {'mesh_file':             input_path + "mesh0.xdmf",\
	# 			   'facets_file':           input_path + "facets0.xdmf",\
	# 			   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf",\
	# 			   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}	
		   	
	# tags = {'intra': 2, 'extra': 1, 'boundary': 4, 'membrane': 2}

	# # CL9 myelin	
	# input_path  = "data/CL9myel/"
	# input_files = {'mesh_file':             input_path + "mesh0.xdmf",\
	# 			   'facets_file':           input_path + "facets0.xdmf",\
	# 			   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf",\
	# 			   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}	
		   	
	# tags = {'intra': 3, 'extra': 1, 'boundary': 4, 'membrane': 2}	
		
	# create EMI problem and ionic model
	problem = EMI_problem(input_files, tags, dt)

	# ionic model	
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)

	# solve
	solver = EMI_solver(problem, time_steps, True, True)	
	solver.solve()

	# HH.plot_svg()

	# mpirun -n 8 python3 -u src/EMIx/EMI/main.py