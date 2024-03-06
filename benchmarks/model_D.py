from EMIx   import *
from dolfin import *

def g_Na_stim(g_syn_bar, a_syn, t):	
	return Expression('10 * (x[0] > 0.00008) * g_syn_bar * exp(-fmod(t,0.01)/a_syn)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)

if __name__=='__main__':
		
	# global time step (s)	
	dt = 0.00005
	time_steps = 100

	# single astocyte in ECS	
	input_files = {'mesh_file':	            "../data/cali/mesh_fn.xdmf", \
				   'facets_file':           "../data/cali/facets_fn.xdmf", \
				   'intra_restriction_dir': "../data/cali/interior_restriction_fn.rtc.xdmf", \
				   'extra_restriction_dir': "../data/cali/exterior_restriction_fn.rtc.xdmf"}
	
	tags = {'intra': (2,3)}

	# create EMI problem and ionic model
	problem = KNPEMI_problem(input_files, tags, dt)	

	# add ionic models
	problem.add_ionic_model("Passive_K_pump", 2)
	problem.add_ionic_model("HH", 3, stim_fun=g_Na_stim)

	# solve with both .xdmf and .png output
	solver = KNPEMI_solver(problem, time_steps, save_xdmf_files=False, save_png_files=True)
	solver.solve()

	
	

	
