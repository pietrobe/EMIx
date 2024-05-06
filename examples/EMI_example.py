from pathlib import Path
from EMIx   import *
from dolfin import *

#  Na stimulus definition
def g_Na_stim(t):			
	return Expression('40*exp(-fmod(t,0.01)/0.002)', t=t, degree=4)				

if __name__=='__main__':
			
	# create EMI problem and ionic model
	problem = EMI_problem('config.yml')	

	# add HH ionic model
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)

	# solve with just .png output
	solver = EMI_solver(problem, save_xdmf_files=False, save_png_files=True)	
	solver.solve()

	

	
	

	
