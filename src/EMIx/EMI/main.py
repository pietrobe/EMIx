from EMI_problem import EMI_problem
from EMI_solver  import EMI_solver 
from dolfin import *
import time
from sys   import argv


#  Na stimulus definition
def g_Na_stim(t):
	
	a_syn     = 0.002           
	g_syn_bar = 40  

	return Expression('g_syn_bar*exp(-fmod(t,0.01)/a_syn)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)				


if __name__=='__main__':
		
	# # global time step (s)	
	# dt = 0.01
	# time_steps = 1

	# # grid size
	# N = int(argv[1])
	
	# # square	
	# input_path  = "../../../data/square/square"
	# input_files = {'mesh_file':             input_path    	                 + str(N) + ".xml", \
	#  			   'subdomais_file': 		input_path + "_physical_region"  + str(N) + ".xml", \
	#  			   'facets_file':           input_path + "_facet_region"     + str(N) + ".xml", \
	#  			   'intra_restriction_dir': input_path + "_restriction_om_i" + str(N) + ".rtc.xml", \
	#  			   'extra_restriction_dir': input_path + "_restriction_om_e" + str(N) + ".rtc.xml"}		

	# # # cube
	# # input_path  = "../../../data/cube/cube_"
	# # input_files = {'mesh_file':             input_path + "regions20.xdmf", \
	# # 			   'facets_file':           input_path + "facets20.xdmf", \
	# # 			   'intra_restriction_dir': input_path + "in_restriction20.rtc.xdmf", \
	# # 			   'extra_restriction_dir': input_path + "ex_restriction20.rtc.xdmf"}
			
	# tags = {'intra': 1 , 'extra': 2, 'boundary': 1, 'membrane': 2}	

	# # Ale myelin	
	# input_path  = "../../../data/myelin/"
	# input_files = {'mesh_file':             input_path + "mesh0.xdmf",\
	# 			   'facets_file':           input_path + "facets0.xdmf",\
	# 			   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf",\
	# 			   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}	
		   	
	# tags = {'intra': 3 , 'extra': 1, 'boundary': 4, 'membrane': 2}	

	# # Ale test	
	# input_path  = "../../../data/Ale_test/"
	# input_files = {'mesh_file':             input_path + "mesh0.xdmf",\
	# 			   'facets_file':           input_path + "facets0.xdmf",\
	# 			   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf",\
	# 			   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}	
		   	
	# tags = {'intra': 2 , 'extra': 1, 'boundary': 4, 'membrane': 2}	

	# Ada test
	# input_path  = "../../../data/Ada_meshes/volume_ncells_200_size_5000/"
	# input_files = {'mesh_file':             input_path + "mesh0.xdmf",\
	#			   'facets_file':           input_path + "facets0.xdmf",\
	#			   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf",\
	#			   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}	
	#	   	
	#intra_tags = tuple(range(2,202))		   	
	#tags = {'intra': intra_tags, 'extra': 1, 'membrane': intra_tags}	
	

	# # many cells
	# N = int(argv[1])
	# K = int(argv[2])
	
	# # square	
	# input_path  = "../../../data/many_cells/squares_"
	# input_files = {'mesh_file':             input_path + "regions" 		  + str(N) + "_" + str(K) + ".xdmf", \
	#   			   'facets_file':           input_path + "facets"  		  + str(N) + "_" + str(K) + ".xdmf", \
	#  			   'intra_restriction_dir': input_path + "in_restriction" + str(N) + "_" + str(K) + ".rtc.xdmf", \
	#   			   'extra_restriction_dir': input_path + "ex_restriction" + str(N) + "_" + str(K) + ".rtc.xdmf"}
	
	# tags = {'intra': 1 , 'extra': 2, 'boundary': 1, 'membrane': 2}				
		
	input_file = "config.yml"

	# create EMI problem and ionic model
	problem = EMI_problem(input_file)

	# ionic model	
	problem.add_ionic_model("Passive")

	# solve
	solver = EMI_solver(problem, False, False)	
	solver.solve()

	# HH.plot_png()

	
	

	
