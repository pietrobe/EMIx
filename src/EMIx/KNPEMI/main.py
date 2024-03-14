from KNPEMI_problem     import KNPEMI_problem 
from KNPEMI_solver      import KNPEMI_solver
from dolfin import *
import time
from sys   import argv

#  Na stimulus definition
def g_Na_stim(g_syn_bar, a_syn, t):			
	return Expression('g_syn_bar*exp(-fmod(t,0.01)/a_syn)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)				


if __name__=='__main__':
		
	# global time step (s)	
	dt = 0.00005
	time_steps = 100

	# grid size
	N = int(argv[1])
	
	# square	
	input_path  = "../../../data/square/"
	input_files = {'mesh_file':             input_path + "square"                  + str(N) + ".xml", \
	 			   'subdomais_file': 		input_path + "square_physical_region"  + str(N) + ".xml", \
	 			   'facets_file':           input_path + "square_facet_region"     + str(N) + ".xml", \
	 			   'intra_restriction_dir': input_path + "square_restriction_om_i" + str(N) + ".rtc.xml", \
	 			   'extra_restriction_dir': input_path + "square_restriction_om_e" + str(N) + ".rtc.xml"}		
	
	# # cube
	# input_path  = "../../../data/cube/"
	# input_files = {'mesh_file':           input_path + "cube_regions" + str(N) + ".xdmf", \
	# 			   'facets_file':           input_path + "cube_facets"  + str(N) + ".xdmf", \
	# 			   'intra_restriction_dir': input_path + "cube_in_restriction" + str(N) + ".rtc.xdmf", \
	# 			   'extra_restriction_dir': input_path + "cube_ex_restriction" + str(N) + ".rtc.xdmf"}
	
	tags = {'intra': 1 , 'extra': 2, 'boundary': 1, 'membrane': 2}	
	
	# # single astocyte in ECS
	# input_path  = "../../../data/astro_in_ECS/offset_2/"
	# input_files = {'mesh_file':	             input_path + "mesh0.xdmf", \
	# 			   'facets_file':            input_path + "facets0.xdmf", \
	# 			   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf", \
	# 			   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}
	
	# tags = {'intra': 2 , 'extra': 1, 'boundary': 4, 'membrane': 3}		

	# # dendrite 
	# input_path  = "../../../data/dendrite/"
	# input_files = {'mesh_file':	            input_path + "mesh.xdmf", \
	# 			   'facets_file':           input_path + "facets.xdmf", \
	# 			   'intra_restriction_dir': input_path + "interior_restriction.rtc.xdmf", \
	# 			   'extra_restriction_dir': input_path + "exterior_restriction.rtc.xdmf"}

	# tags = {'intra': (2,3,4) , 'extra': 1, 'boundary': 1, 'membrane': (2,3,4)}
		
	# # Marius
	# path = "../../../data/marius_meshes/volume_ncells_5_size_5000/"
	# input_files = [ path + "mesh_out.xdmf", "", path + "facets.xdmf", \
	# 				path + "interior_restriction.rtc.xdmf", path + "exterior_restriction.rtc.xdmf", ""]		

	# # tags for intra, extra, bound, gamma
	# tags = [(2,3,4,5,6), 1, 4, 3] 

	# # Ale myelin	
	# input_path  = "../../../data/Ale_test/"
	# input_files = {'mesh_file':             input_path + "test.xdmf",\
	# 			   'facets_file':           input_path + "facets.xdmf",\
	# 			   'intra_restriction_dir': input_path + "interior_restriction.rtc.xdmf",\
	# 			   'extra_restriction_dir': input_path + "exterior_restriction.rtc.xdmf"}	
		   	
	# tags = {'intra': 3 , 'extra': 1, 'boundary': 4, 'membrane': 2}	


	# create KNP-EMI problem and solver
	problem = KNPEMI_problem(input_files, tags, dt)		
	
	# set ionic models
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)
	
	# solve
	solver = KNPEMI_solver(problem, time_steps, False, True)
	solver.solve()


