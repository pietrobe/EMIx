from EMIx import *

if __name__=='__main__':
		
	# global time step (s)	
	dt = 0.0001
	time_steps = 100

	# single astocyte in ECS	
	input_files = {'mesh_file':	            "../data/astro_in_ECS/offset_2/mesh0.xdmf", \
				   'facets_file':           "../data/astro_in_ECS/offset_2/facets0.xdmf", \
				   'intra_restriction_dir': "../data/astro_in_ECS/offset_2/interior_restriction0.rtc.xdmf", \
				   'extra_restriction_dir': "../data/astro_in_ECS/offset_2/exterior_restriction0.rtc.xdmf"}
	
	tags = {'intra': 2 , 'extra': 1, 'boundary': 4, 'membrane': 3}		

	# create EMI problem and ionic model
	problem = KNPEMI_problem(input_files, tags, dt)	

	# set sources
	problem.Na_e_f = Expression('(x[0] < 0.00033) * (t <= 0.002)', t=t, degree=1)				
	problem.K_e_f  = Expression('(x[0] < 0.00033) * (t <= 0.002)', t=t, degree=1)				
	
	# add passive ionic model
	problem.add_ionic_model("Passive_K_pump")

	# solve with both .xdmf and .png output
	solver = KNPEMI_solver(problem, time_steps, save_xdmf_files=False, save_png_files=True)
	solver.solve()

	
	

	
