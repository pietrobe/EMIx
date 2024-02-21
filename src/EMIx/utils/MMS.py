from dolfin import *
from sympy.printing import ccode
from dolfin         import errornorm
import numpy        as np
import sympy        as sp


class setup_MMS:
    """ class for calculating source terms of the KNP-EMI system for given exact
    solutions """
    def __init__(self):
        # define symbolic variables
        self.x, self.y, self.t = sp.symbols('x[0] x[1] t')
        return

    def get_exact_solution(self):
        """ define manufactured (exact) solutions """
        x = self.x; y = self.y; t = self.t

        # sodium (Na) concentration
        Na_i_e = 0.7 + 0.3*sp.sin(2*pi*x)*sp.sin(2*pi*y)*sp.exp(-t)
        Na_e_e = 1.0 + 0.6*sp.sin(2*pi*x)*sp.sin(2*pi*y)*sp.exp(-t)
        # potassium (K) concentration
        K_i_e = 0.3 + 0.3*sp.sin(2*pi*x)*sp.sin(2*pi*y)*sp.exp(-t)
        K_e_e = 1.0 + 0.2*sp.sin(2*pi*x)*sp.sin(2*pi*y)*sp.exp(-t)
        # chloride (Cl) concentration
        Cl_i_e = 1.0 + 0.6*sp.sin(2*pi*x)*sp.sin(2*pi*y)*sp.exp(-t)
        Cl_e_e = 2.0 + 0.8*sp.sin(2*pi*x)*sp.sin(2*pi*y)*sp.exp(-t)
        # potentials
        phi_i_e = sp.cos(2*pi*x)*sp.cos(2*pi*y)*(1 + sp.exp(-t))
        phi_e_e = sp.cos(2*pi*x)*sp.cos(2*pi*y)

        exact_solutions = {'Na_i_e':Na_i_e, 'K_i_e':K_i_e, 'Cl_i_e':Cl_i_e,\
                           'Na_e_e':Na_e_e, 'K_e_e':K_e_e, 'Cl_e_e':Cl_e_e,\
                           'phi_i_e':phi_i_e, 'phi_e_e':phi_e_e}
        return exact_solutions

    def get_MMS_terms_KNPEMI(self, time):
        """ get exact solutions, source terms, boundary terms and initial
            conditions for the method of manufactured solution (MMS) for the
            KNP-EMI problem """
        x = self.x; y = self.y; t = self.t
        # get manufactured solution
        exact_solutions = self.get_exact_solution()
        # unwrap exact solutions
        for key in exact_solutions:
            # exec() changed from python2 to python3
            exec('global %s; %s = exact_solutions["%s"]' % (key, key ,key))

        # Calculate components
        # gradients
        grad_Nai, grad_Ki, grad_Cli, grad_phii, grad_Nae, grad_Ke, grad_Cle, grad_phie = \
                [np.array([sp.diff(foo, x),  sp.diff(foo, y)])
                for foo in (Na_i_e, K_i_e, Cl_i_e, phi_i_e, Na_e_e, K_e_e, Cl_e_e, phi_e_e)]

        # compartmental fluxes
        J_Na_i = - grad_Nai - Na_i_e*grad_phii
        J_Na_e = - grad_Nae - Na_e_e*grad_phie
        J_K_i  = - grad_Ki  - K_i_e*grad_phii
        J_K_e  = - grad_Ke  - K_e_e*grad_phie
        J_Cl_i = - grad_Cli + Cl_i_e*grad_phii
        J_Cl_e = - grad_Cle + Cl_e_e*grad_phie

        # membrane potential
        phi_M_e = phi_i_e - phi_e_e

        # total membrane flux defined intracellularly (F sum_k z^k J^k_i)
        total_flux_i = J_Na_i + J_K_i - J_Cl_i
        # current defined intracellular: total_flux_i dot i_normals
        # [(-1, 0), (1, 0), (0, -1), (0, 1)]
        JMe_i = [- total_flux_i[0], total_flux_i[0], - total_flux_i[1], total_flux_i[1]]

        # membrane flux defined extracellularly ( - F sum_k z^k J^k_i)
        total_flux_e = - (J_Na_e + J_K_e - J_Cl_e)
        # current defined intracellular: total_flux_e dot e_normals
        # [(1, 0), (-1, 0), (0, 1), (0, -1)]
        JMe_e = [total_flux_e[0], - total_flux_e[0], total_flux_e[1], - total_flux_e[1]]

        # ion channel currents
        I_ch_Na = phi_M_e                 # Na
        I_ch_K  = phi_M_e                 # K
        I_ch_Cl = phi_M_e                 # Cl
        I_ch = I_ch_Na + I_ch_K + I_ch_Cl # total

        # setup subdomain for internal and external facets - normals (1, 0), (1, 0), (0, 1), (0, 1)
        subdomains_MMS = [('near(x[0], 0)', 'near(x[0], 1)', 'near(x[1], 0)', 'near(x[1], 1)'),
                          ('near(x[0], 0.25)', 'near(x[0], 0.75)', 'near(x[1], 0.25)', 'near(x[1], 0.75)')]

        # Calculate source terms
        # equations for ion cons: f = dk_r/dt + div (J_kr)
        f_Na_i = sp.diff(Na_i_e, t) + sp.diff(J_Na_i[0], x) + sp.diff(J_Na_i[1], y)
        f_Na_e = sp.diff(Na_e_e, t) + sp.diff(J_Na_e[0], x) + sp.diff(J_Na_e[1], y)
        f_K_i  = sp.diff(K_i_e, t)  + sp.diff(J_K_i[0], x)  + sp.diff(J_K_i[1], y)
        f_K_e  = sp.diff(K_e_e, t)  + sp.diff(J_K_e[0], x)  + sp.diff(J_K_e[1], y)
        f_Cl_i = sp.diff(Cl_i_e, t) + sp.diff(J_Cl_i[0], x) + sp.diff(J_Cl_i[1], y)
        f_Cl_e = sp.diff(Cl_e_e, t) + sp.diff(J_Cl_e[0], x) + sp.diff(J_Cl_e[1], y)

        # equations for potentials: fE = - F sum(z_k*div(J_k_r)
        f_phi_i = - ((sp.diff(J_Na_i[0], x) + sp.diff(J_Na_i[1], y))
                  + ( sp.diff(J_K_i[0],  x) + sp.diff(J_K_i[1],  y))
                  - ( sp.diff(J_Cl_i[0], x) + sp.diff(J_Cl_i[1], y)))
        f_phi_e = - ((sp.diff(J_Na_e[0], x) + sp.diff(J_Na_e[1], y))
                  + ( sp.diff(J_K_e[0],  x) + sp.diff(J_K_e[1],  y))
                  - ( sp.diff(J_Cl_e[0], x) + sp.diff(J_Cl_e[1], y)))

        # equation for phi_M: f = C_M*d(phi_M)/dt - (I_M - I_ch) where we have
        # chosen I_M = F sum_k z^k J_i^k n_i = total_flux_i
        fJM = [sp.diff(phi_M_e, t) + I_ch - foo for foo in JMe_i]
        # coupling condition for I_M: (total_flux_i*n_i) = (- total_flux_e*n_e) + f
        # giving f = total_flux_i*n_i + total_flux_e*n_e
        fgM = [i - e for i, e in zip(JMe_i, JMe_e)]

        # Convert to expressions
        # exact solutions
        Nai_e, Nae_e, Ki_e, Ke_e, Cli_e, Cle_e, phii_e, phie_e, phiM_e = \
                [Expression(sp.printing.ccode(foo), t=time, degree=4)
                for foo in (Na_i_e, Na_e_e, K_i_e, K_e_e, Cl_i_e, Cl_e_e, phi_i_e, phi_e_e, phi_M_e)]

        # exact membrane flux
        JM_e = [Expression(sp.printing.ccode(foo), t=time, degree=4) for foo in JMe_i]

        # source terms
        f_Nai, f_Nae, f_Ki, f_Ke, f_Cli, f_Cle, f_phii, f_phie = \
                [Expression(sp.printing.ccode(foo), t=time, degree=4)
                for foo in (f_Na_i, f_Na_e, f_K_i, f_K_e, f_Cl_i, f_Cl_e, f_phi_i, f_phi_e)]

        # source term membrane flux
        f_JM = [Expression(sp.printing.ccode(foo), t=time, degree=4) for foo in fJM]

        # source term continuity coupling condition on gamma
        f_gM = [Expression(sp.printing.ccode(foo), t=time, degree=4) for foo in fgM]

        # initial conditions concentrations
        init_Nai, init_Nae, init_Ki, init_Ke, init_Cli, init_Cle, init_phiM = \
                [Expression(sp.printing.ccode(foo), t=time, degree=4)
                for foo in (Na_i_e, Na_e_e, K_i_e, K_e_e, Cl_i_e, Cl_e_e, phi_M_e)]

        # exterior boundary terms
        J_Nae, J_Ke, J_Cle = [Expression((sp.printing.ccode(foo[0]),
                             sp.printing.ccode(foo[1])), t=time, degree=4)
                             for foo in (J_Na_e, J_K_e, J_Cl_e)]

        # ion channel currents
        I_ch_Na, I_ch_K, I_ch_Cl = \
                [Expression(sp.printing.ccode(foo), t=time, degree=4)
                 for foo in (I_ch_Na, I_ch_K, I_ch_Cl)]

        # Gather expressions
        # exact solutions
        exact_sols = {'Na_i_e':Nai_e, 'K_i_e':Ki_e, 'Cl_i_e':Cli_e,
                      'Na_e_e':Nae_e, 'K_e_e':Ke_e, 'Cl_e_e':Cle_e,
                      'phi_i_e':phii_e, 'phi_e_e':phie_e, 'phi_M_e':phiM_e,
                      'I_M_e':JM_e, 'I_ch_Na':I_ch_Na, 'I_ch_K':I_ch_K,
                      'I_ch_Cl':I_ch_Cl}
        # source terms
        src_terms = {'f_Na_i':f_Nai, 'f_K_i':f_Ki, 'f_Cl_i':f_Cli,
                     'f_Na_e':f_Nae, 'f_K_e':f_Ke, 'f_Cl_e':f_Cle,
                     'f_phi_i':f_phii, 'f_phi_e':f_phie, 'f_I_M':f_JM,
                     'f_g_M':f_gM}
        # initial conditions
        init_conds = {'Na_i':init_Nai, 'K_i':init_Ki, 'Cl_i':init_Cli,
                      'Na_e':init_Nae, 'K_e':init_Ke, 'Cl_e':init_Cle,
                      'phi_M':init_phiM}
        # boundary terms
        bndry_terms = {'J_Na_e':J_Nae, 'J_K_e':J_Ke, 'J_Cl_e':J_Cle}
        return src_terms, exact_sols, init_conds, bndry_terms, subdomains_MMS
