from EMIx   import *
from dolfin import *

#  Na stimulus definition
def g_Na_stim(g_syn_bar, a_syn, t):			
	return Expression('2*g_syn_bar', g_syn_bar=g_syn_bar, degree=1)			

if __name__=='__main__':
		
	# global time step (s)	
	dt = 0.00005
	time_steps = 100

	# dendrite 	
	input_files = {'mesh_file':	            "../data/dendrite/mesh.xdmf", \
				   'facets_file':           "../data/dendrite/facets.xdmf", \
				   'intra_restriction_dir': "../data/dendrite/interior_restriction.rtc.xdmf", \
				   'extra_restriction_dir': "../data/dendrite/exterior_restriction.rtc.xdmf"}

	# NOTE: by default extra_tag = 1 and membrane_tags = intra_tags
	tags = {'intra': (2,3,4)}

	# create KNPEMI problem and ionic model
	problem = KNPEMI_problem(input_files, tags, dt)		
	
	# add models depending on membrane tags

	# astrocytes
	problem.add_ionic_model("Passive_K_pump", 2)                     

	# dendrite heads
	problem.add_ionic_model("Passive_Nerst",  3, stim_fun=g_Na_stim) 
	
	# dendrite
	problem.add_ionic_model("Passive_Nerst",  4)				     
	
	# solve with both .xdmf and .png output
	solver = KNPEMI_solver(problem, time_steps, save_xdmf_files=False, save_png_files=True)
	solver.solve()

	
	

	
