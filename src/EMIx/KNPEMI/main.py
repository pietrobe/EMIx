from KNPEMI_problem     import KNPEMI_problem 
from KNPEMI_solver      import KNPEMI_solver
from dolfin import *
import time
from sys   import argv

#  Na stimulus definition
def g_Na_stim(t):		

	# stimulus
	a_syn     = 0.002           
	g_syn_bar = 40    

	return Expression('g_syn_bar*exp(-fmod(t,0.01)/a_syn)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)				


if __name__=='__main__':		

	input_file = "config.yml"

	# create KNP-EMI problem and solver
	problem = KNPEMI_problem(input_file)			
	
	# set ionic models
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)
	
	# solve
	solver = KNPEMI_solver(problem, save_xdmf_files=True, save_png_files=True)
	solver.solve()


