from EMIx    import *
from dolfin  import *
from pathlib import Path

# final membrane potential (hardcoded)
FINAL_PHI_M = 30.30931388

#  Na stimulus definition
def g_Na_stim(g_syn_bar, a_syn, t):			
	return Expression('g_syn_bar*exp(-fmod(t,0.01)/a_syn)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)				

if __name__=='__main__':
			
	# input files
	input_dir   = (Path(__file__).parent.parent.parent / "data/square").absolute().as_posix()	
	input_files = {'mesh_file':             input_dir + "/square32.xml", \
		 		   'subdomais_file': 		input_dir + "/square_physical_region32.xml", \
		 		   'facets_file':           input_dir + "/square_facet_region32.xml", \
		 		   'intra_restriction_dir': input_dir + "/square_restriction_om_i32.rtc.xml", \
		 		   'extra_restriction_dir': input_dir + "/square_restriction_om_e32.rtc.xml"}		
	
	# NOTE: boundary tag is not necessary for Neumann BC		   			   		
	tags = {'intra': 1 , 'extra': 2, 'membrane': 2}	
		
	# create EMI problem and ionic model
	problem = EMI_problem(input_files, tags, dt=0.00002)	

	# add HH ionic model
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)

	# solve with just .png output
	solver = EMI_solver(problem, time_steps=50, save_xdmf_files=False, save_png_files=True)	
	solver.solve()
	
	# test
	final_err_v = abs(solver.v_t[-1] - FINAL_PHI_M)
	print(final_err_v)
	assert final_err_v < solver.ksp_rtol


	

	
	

	
