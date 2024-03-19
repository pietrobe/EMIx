from EMIx   import *
from dolfin import *


# values from direct solve at t = T
FINAL_PHI_M = 36.66709614

MAX_FINAL_NA_i  = 12.378971866657572
MAX_FINAL_NA_e  = 99.89956401593034
MAX_FINAL_K_i   = 124.72211191441083
MAX_FINAL_K_e   = 4.073318584067195
MAX_FINAL_CL_i  = 137.09026107750998
MAX_FINAL_CL_e  = 103.96576469612945

# tolerances for direct and iterative
TOL_DIR = 1e-10
TOL_KSP = 1e-1

#  Na stimulus definition
def g_Na_stim(g_syn_bar, a_syn, t):			
	return Expression('g_syn_bar*exp(-fmod(t,0.01)/a_syn)', g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)				

if __name__=='__main__':
			
	# input files	
	input_files = {'mesh_file':             "../../data/square/square32.xml", \
		 		   'subdomais_file': 		"../../data/square/square_physical_region32.xml", \
		 		   'facets_file':           "../../data/square/square_facet_region32.xml", \
		 		   'intra_restriction_dir': "../../data/square/square_restriction_om_i32.rtc.xml", \
		 		   'extra_restriction_dir': "../../data/square/square_restriction_om_e32.rtc.xml"}		
	   	
	# NOTE: boundary tag is not necessary for Neumann BC		   	
	tags = {'intra': 1 , 'extra': 2, 'membrane': 2}	
		
	# create EMI problem and ionic model
	problem = KNPEMI_problem(input_files, tags, dt=0.00002)		

	# add HH ionic model
	problem.add_ionic_model("HH", stim_fun=g_Na_stim)

	# solve with both .xdmf and .png output
	solver = KNPEMI_solver(problem, time_steps=50, save_xdmf_files=False, save_png_files=True)	

	solver.direct_solver = True
	solver.solve()

	# testing direct solve

	# membrane potential
	final_err_v = abs(solver.v_t[-1] - FINAL_PHI_M)	
	print(final_err_v)
	assert final_err_v < 1e-4

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

	# print(err_Na_i)
	# print(err_Na_e)
	# print(err_K_i)
	# print(err_K_e)
	# print(err_Cl_i)
	# print(err_Cl_e)

	# tests
	assert err_Na_i < TOL_DIR
	assert err_Na_e < TOL_DIR
	assert err_K_i  < TOL_DIR
	assert err_K_e  < TOL_DIR
	assert err_Cl_i < TOL_DIR
	assert err_Cl_e < TOL_DIR

	# testing iterative
	problem.t = Constant(0.0)
	solver.direct_solver  = False
	solver.save_pngs      = False
	solver.solve()
	
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

	# print(err_Na_i)
	# print(err_Na_e)
	# print(err_K_i)
	# print(err_K_e)
	# print(err_Cl_i)
	# print(err_Cl_e)

	# tests
	assert err_Na_i < TOL_KSP
	assert err_Na_e < TOL_KSP
	assert err_K_i  < TOL_KSP
	assert err_K_e  < TOL_KSP
	assert err_Cl_i < TOL_KSP
	assert err_Cl_e < TOL_KSP





	
	
