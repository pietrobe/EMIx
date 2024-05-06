from EMIx   import *
from dolfin import *

def g_Na_stim(t):	
	return Expression('400 * (x[0] > 0.00008) * exp(-fmod(t,0.01)/0.002)', t=t, degree=4)

if __name__=='__main__':
				
	# create EMI problem and ionic model
	problem = KNPEMI_problem('config_D.yml')	

	# add ionic models
	problem.add_ionic_model("Passive_K_pump", 2)
	problem.add_ionic_model("HH", 3, stim_fun=g_Na_stim)

	# solve with both .xdmf and .png output
	solver = KNPEMI_solver(problem, save_xdmf_files=False, save_png_files=True)
	solver.solve()

	
	

	
