from EMIx.KNPEMI.KNPEMI_problem     import KNPEMI_problem
from EMIx.KNPEMI.KNPEMI_solver      import KNPEMI_solver 
from EMIx.KNPEMI.KNPEMI_ionic_model import *
import time
from sys   import argv

if __name__=='__main__':
		
	# global time step (s)	
	dt = 0.00002
	time_steps = 10

	# input files	
	input_path  = "../data/myelin/"
	input_files = {'mesh_file':             input_path + "mesh0.xdmf",\
				   'facets_file':           input_path + "facets0.xdmf",\
				   'intra_restriction_dir': input_path + "interior_restriction0.rtc.xdmf",\
				   'extra_restriction_dir': input_path + "exterior_restriction0.rtc.xdmf"}	
		   	
	tags = {'intra': 3 , 'extra': 1, 'boundary': 4, 'membrane': 2}	
		
	# create EMI problem and ionic model
	problem = KNPEMI_problem(input_files, tags, dt)	

	HH = HH_model(problem)	
	problem.init_ionic_model([HH])

	# solve
	solver = KNPEMI_solver(problem, time_steps)
	solver.solve()

	
	

	
