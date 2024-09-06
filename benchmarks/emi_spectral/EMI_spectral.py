from pathlib import Path
from EMIx   import *
from dolfin import *

if __name__=='__main__':
			
	# create EMI problem and ionic model
	problem = EMI_problem('config.yml')	

	# add HH ionic model
	problem.add_ionic_model("Passive")

	# solve with just .png output
	solver = EMI_solver(problem, save_xdmf_files=True,
						save_png_files=True)	
	solver.solve()

	

	
	

	
