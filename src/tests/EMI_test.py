from EMIx    import *
from dolfin  import *
from pathlib import Path

# final membrane potential (hardcoded)
FINAL_PHI_M = 30.30929903

#  Na stimulus definition
def g_Na_stim(t):		
		
	return Expression('40*exp(-fmod(t,0.01)/0.002 )', t=t, degree=4)				

if __name__=='__main__':
					
	# create EMI problem and ionic model
	problem = EMI_problem('config.yml')	

	# add HH ionic model
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)

	# solve with just .png output
	solver = EMI_solver(problem, save_xdmf_files=False, save_png_files=True)	
	solver.use_direct = True
	solver.solve()
	
	# test
	final_err_v = abs(solver.v_t[-1] - FINAL_PHI_M)
	print(final_err_v)
	assert final_err_v < solver.ksp_rtol


	

	
	

	
