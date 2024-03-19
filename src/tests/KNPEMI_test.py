from EMIx   import *
from dolfin import *

#  Na stimulus definition
def g_Na_stim(g_syn_bar, a_syn, t):			
	return Expression('g_syn_bar*exp(-fmod(t,0.01)/a_syn)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)				

if __name__=='__main__':
		
	# global time step (s)	
	dt = 0.00002
	time_steps = 50

	# input files	
	input_files = {'mesh_file':             "../../data/square/square32.xml", \
		 		   'subdomais_file': 		"../../data/square/square_physical_region32.xml", \
		 		   'facets_file':           "../../data/square/square_facet_region32.xml", \
		 		   'intra_restriction_dir': "../../data/square/square_restriction_om_i32.rtc.xml", \
		 		   'extra_restriction_dir': "../../data/square/square_restriction_om_e32.rtc.xml"}		
	   	
	# NOTE: boundary tag is not necessary for Neumann BC		   	
	tags = {'intra': 1 , 'extra': 2, 'membrane': 2}	
		
	# create EMI problem and ionic model
	problem = KNPEMI_problem(input_files, tags, dt)		

	# add HH ionic model
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)

	# solve with both .xdmf and .png output
	solver = KNPEMI_solver(problem, time_steps)
	solver.solve()

	
	

	
