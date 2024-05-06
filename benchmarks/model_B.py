from EMIx import *

if __name__=='__main__':
				
	# create EMI problem and ionic model
	problem = KNPEMI_problem('config_B.yml')	

	# set sources
	problem.Na_e_f = Expression('(x[0] < 0.00033) * (t <= 0.002)', t=t, degree=1)				
	problem.K_e_f  = Expression('(x[0] < 0.00033) * (t <= 0.002)', t=t, degree=1)				
	
	# add passive ionic model
	problem.add_ionic_model("Passive_K_pump")

	# solve with both .xdmf and .png output
	solver = KNPEMI_solver(problem, save_xdmf_files=False, save_png_files=True)
	solver.solve()

	
	

	
