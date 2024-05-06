from EMIx    import *
from dolfin  import *
from pathlib import Path

# values from direct solve at t = T
FINAL_PHI_M = 36.66709614

MAX_FINAL_NA_i  = 12.378971866657572
MAX_FINAL_NA_e  = 99.89956401593034
MAX_FINAL_K_i   = 124.72211191441083
MAX_FINAL_K_e   = 4.073318584067195
MAX_FINAL_CL_i  = 137.09026107750998
MAX_FINAL_CL_e  = 103.96576469612945

# tolerances for direct and iterative
TOL       = 1e-9
TOL_PHI_M = 1e-4

#  Na stimulus definition
def g_Na_stim(t):			
	return Expression('40*exp(-fmod(t,0.01)/0.002)', t=t, degree=4)				

if __name__=='__main__':
			
	# # input files	
	# input_dir   = (Path(__file__).parent.parent.parent / "data/square").absolute().as_posix()	
	# input_files = {'mesh_file':             input_dir + "/square32.xml", \
	# 	 		   'subdomais_file': 		input_dir + "/square_physical_region32.xml", \
	# 	 		   'facets_file':           input_dir + "/square_facet_region32.xml", \
	# 	 		   'intra_restriction_dir': input_dir + "/square_restriction_om_i32.rtc.xml", \
	# 	 		   'extra_restriction_dir': input_dir + "/square_restriction_om_e32.rtc.xml"}		
		   	
	# # NOTE: boundary tag is not necessary for Neumann BC		   	
	# tags = {'intra': 1 , 'extra': 2, 'membrane': 2}	
		
	# create EMI problem and ionic model
	problem = KNPEMI_problem('config.yml')		

	# add HH ionic model
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)

	# solve with both .xdmf and .png output
	solver = KNPEMI_solver(problem, save_xdmf_files=False, save_png_files=True)	
	solver.direct_solver = True
	solver.solve()

	# testing direct solve

	# membrane potential
	final_err_v = abs(solver.v_t[-1] - FINAL_PHI_M)	
	
	# concentrations
	ui = problem.wh.sub(0)
	ue = problem.wh.sub(1)

	# Na	
	ui_Na_max = ui.sub(0, deepcopy=True).vector().max()
	ue_Na_max = ue.sub(0, deepcopy=True).vector().max()

	# K
	ui_K_max = ui.sub(1, deepcopy=True).vector().max()	
	ue_K_max = ue.sub(1, deepcopy=True).vector().max()

	# Cl	
	ui_Cl_max = ui.sub(2, deepcopy=True).vector().max()	
	ue_Cl_max = ue.sub(2, deepcopy=True).vector().max()

	err_Na_i = abs(MAX_FINAL_NA_i - ui_Na_max)/abs(ui_Na_max)
	err_K_i  = abs(MAX_FINAL_K_i  - ui_K_max) /abs(ui_K_max)
	err_Cl_i = abs(MAX_FINAL_CL_i - ui_Cl_max)/abs(ui_Cl_max)
	err_Na_e = abs(MAX_FINAL_NA_e - ue_Na_max)/abs(ue_Na_max)
	err_K_e  = abs(MAX_FINAL_K_e  - ue_K_max) /abs(ue_K_max)
	err_Cl_e = abs(MAX_FINAL_CL_e - ue_Cl_max)/abs(ue_Cl_max)

	print('\nErrors:')
	print(final_err_v)
	print('-------------')
	print(err_Na_i)
	print(err_Na_e)
	print(err_K_i)
	print(err_K_e)
	print(err_Cl_i)
	print(err_Cl_e)

	# tests
	assert final_err_v < TOL_PHI_M
	assert err_Na_i < TOL
	assert err_Na_e < TOL
	assert err_K_i  < TOL
	assert err_K_e  < TOL
	assert err_Cl_i < TOL
	assert err_Cl_e < TOL