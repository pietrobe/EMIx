from pathlib import Path
from EMIx   import *
from dolfin import *
import argparse

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Run EMI solver')
	parser.add_argument('configfile', type=str, help='the config ile')
	args = parser.parse_args()

	# create EMI problem and ionic model
	problem = EMI_problem(args.configfile)	

	# add HH ionic model
	problem.add_ionic_model("Passive")

	# solve with just .png output
	solver = EMI_solver(problem, save_xdmf_files=True,
						save_png_files=True)	
	solver.solve()

	

	
	

	
