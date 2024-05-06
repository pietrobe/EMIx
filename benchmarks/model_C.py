from EMIx   import *
from dolfin import *

#  Na stimulus definition
def g_Na_stim(t):			
	return Expression('80', degree=1)			

if __name__=='__main__':
			
	# create KNPEMI problem and ionic model
	problem = KNPEMI_problem('config_C.yml')		
	
	# add models depending on membrane tags

	# astrocytes
	problem.add_ionic_model("Passive_K_pump", 2)                     

	# dendrite heads
	problem.add_ionic_model("Passive_Nerst",  3, stim_fun=g_Na_stim) 
	
	# dendrite
	problem.add_ionic_model("Passive_Nerst",  4)				     
	
	# solve with both .xdmf and .png output
	solver = KNPEMI_solver(problem, save_xdmf_files=False, save_png_files=True)
	solver.solve()

	
	

	
