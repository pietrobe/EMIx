from EMIx   import *
from dolfin import *

#  Na stimulus definition
def g_Na_stim(g_syn_bar, a_syn, t):			
	return Expression('g_syn_bar*exp(-fmod(t,0.01)/a_syn)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)				

if __name__=='__main__':
		
	# global time step (s)	
	dt = 0.00002
	time_steps = 100

	# input files	
	input_path  = "../data/myelin/"
	input_files = {'mesh_file':             input_path + "mesh0.xdmf",\
				   'facets_file':           input_path + "facets0.xdmf",\
				   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf",\
				   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}	
		   	
	# NOTE: boundary tag is not necessary for Neumann BC		   			   		
	tags = {'intra': 3 , 'extra': 1, 'membrane': 2}	
		
	# create EMI problem and ionic model
	problem = EMI_problem(input_files, tags, dt)	

	# add HH ionic model
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)

	# solve with just .png output
	solver = EMI_solver(problem, time_steps, save_xdmf_files=False, save_png_files=True)	
	solver.solve()

	

	
	

	
