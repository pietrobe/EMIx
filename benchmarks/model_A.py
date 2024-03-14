from EMIx   import *
from dolfin import *

#  Na stimulus definition
def g_Na_stim(g_syn_bar, a_syn, t):			
	return Expression('g_syn_bar*exp(-fmod(t,0.01)/a_syn)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)				


if __name__=='__main__':
		
	# temporal discretization
	dt = 0.00005
	time_steps = 10

	# spatial discretization
	N_x = 128
	p = 1

	# dimension
	d = 2
	
	# square	
	if d == 2:
				
		input_files = {'mesh_file':             "../data/square/square"                  + str(N_x) + ".xml", \
		 			   'subdomais_file': 		"../data/square/square_physical_region"  + str(N_x) + ".xml", \
		 			   'facets_file':           "../data/square/square_facet_region"     + str(N_x) + ".xml", \
		 			   'intra_restriction_dir': "../data/square/square_restriction_om_i" + str(N_x) + ".rtc.xml", \
		 			   'extra_restriction_dir': "../data/square/square_restriction_om_e" + str(N_x) + ".rtc.xml"}		
	# cube
	elif d == 3:	 			   
		
		input_files = {'mesh_file':             "../data/cube/cube_regions"        + str(N_x) + ".xdmf", \
					   'facets_file':           "../data/cube/cube_facets"         + str(N_x) + ".xdmf", \
					   'intra_restriction_dir': "../data/cube/cube_in_restriction" + str(N_x) + ".rtc.xdmf", \
					   'extra_restriction_dir': "../data/cube/cube_ex_restriction" + str(N_x) + ".rtc.xdmf"}

	else:
		print("d should be in {2,3}")					   
		exit()
	
	tags = {'intra': 1 , 'extra': 2, 'membrane': 2}	

	# create EMI problem and ionic model
	problem = KNPEMI_problem(input_files, tags, dt)	
	problem.fem_order = p

	# add HH ionic model with stimuls
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)

	# solve with both .xdmf and .png output
	solver = KNPEMI_solver(problem, time_steps, save_xdmf_files=False, save_png_files=False)
	solver.solve()

	
	

	
