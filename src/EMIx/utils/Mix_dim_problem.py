from dolfin import *
from abc    import ABC, abstractmethod
from EMIx.utils.misc import check_if_file_exists
from multiphenics   import *
import collections.abc
import time
import yaml

def flatten_list(input_list):
    return [item for sublist in input_list for item in (sublist if isinstance(sublist, tuple) else [sublist])]


class Mixed_dimensional_problem(ABC):    

    t = Constant(0.0) 
        
    def __init__(self, config_file):  
        
        if MPI.comm_world.rank == 0: print("Reading input data from:", config_file)
        tic = time.perf_counter() 

        # read parametrs from input config file 
        self.read_config_file(config_file)             
                
        # in case some problem dependent init is needed
        self.init()
                        
        # setup FEM
        self.setup_domain()  
        self.setup_spaces() 
        self.setup_boundary_conditions()        

        # init empty ionic model (to be filled)
        self.ionic_models = []      

        if MPI.comm_world.rank == 0: print(f"Problem setup in {time.perf_counter() - tic:0.4f} seconds\n")   


    def read_config_file(self, config_file):
        
        # read input yml file
        with open(config_file, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)        
            
        # set input files and tags
        self.input_files = dict()

        if 'input_dir' in config:
            input_dir = config['input_dir']
        else:         
            input_dir = ''

        if 'cell_tag_file' in config and 'facet_tag_file' in config:  

            mesh_file  = input_dir + config['cell_tag_file']
            facet_file = input_dir + config['facet_tag_file']

            check_if_file_exists(mesh_file)
            check_if_file_exists(facet_file)

            self.input_files['mesh_file']   = mesh_file
            self.input_files['facets_file'] = facet_file
        else:
            print('Provide cell_tag_file and facet_tag_file fields in input .yml file')
            return

        if 'intra_restriction_dir' in config and 'extra_restriction_dir' in config:  

            restriction_i  = input_dir + config['intra_restriction_dir']
            restriction_e  = input_dir + config['extra_restriction_dir']

            check_if_file_exists(restriction_i)
            check_if_file_exists(restriction_e)

            self.input_files['intra_restriction_dir'] = restriction_i
            self.input_files['extra_restriction_dir'] = restriction_e
        else:
            print('Provide restrictions directories in input .yml file')
            return        

        # init time step
        if 'dt' in config:
            self.dt = Constant(config['dt'])        
        else:
            print('Provide dt in input file')
            return

        if 'time_steps' in config: 
            self.time_steps = config['time_steps']            
        elif 'T' in config:            
            self.time_steps = int(config['T']/config['dt'])        
        else:
            print('ERROR: provide final time T or time_steps in input .yml file!')
            exit()

        # set tags
        tags = dict()
        tags['intra'] = config['ics_tags']

        if 'ecs_tags' in config:
            tags['extra'] = config['ecs_tags']

        if 'boundary_tags' in config:
            tags['boundary'] = config['boundary_tags']

        if 'membrane_tags' in config:
            tags['membrane'] = config['membrane_tags']

        # parse tags
        self.parse_tags(tags)        

        # set physical parameters
        if 'physical_constants' in config:
            physical_const = config['physical_constants']
            
            if 'T' in physical_const: self.T = physical_const['T']
            if 'R' in physical_const: self.R = physical_const['R']
            if 'F' in physical_const: self.F = physical_const['F']                        
            self.psi = self.R*self.T/self.F    
        
        if 'C_M' in config: self.C_M = config['C_M']
            
        # scaling mesh factor (dafult 1)
        if 'mesh_conversion_factor' in config: self.m_conversion_factor = config['mesh_conversion_factor']
        
        # finite element polynomial order (dafult 1) 
        if 'fem_order' in config: self.fem_order = config['fem_order']

        # boundary conditions (dafult False)
        if 'dirichlet_bcs' in config: self.dirichlet_bcs = config['dirichlet_bcs']      

        # initial membrane potential
        if 'phi_M_init' in config: self.phi_M_init = Constant(config['phi_M_init'])

        # set diffusivities (for EMI)
        if 'sigma_i' in config: self.sigma_i = config['sigma_i']      
        if 'sigma_e' in config: self.sigma_e = config['sigma_e']      
    
        # set parameters of ions (for KNP-EMI)
        if'ion_species' in config:

            self.ion_list = []

            for ion in config['ion_species']:
                
                ion_dict = {'name':ion}   

                ion_params = config['ion_species'][ion]                   

                # safety checks
                if 'valence' not in ion_params: 
                    print('ERROR: valence of ', ion, 'should be provided!')
                    return
                if 'diffusivity' not in ion_params: 
                    print('ERROR: diffusivity of ', ion, 'should be provided!')
                    return
                if 'initial' not in ion_params: 
                    print('ERROR: initial of ', ion, 'should be provided!')
                    return

                # fill ion information
                ion_dict['z']       = ion_params['valence']                                
                ion_dict['Di']      = Constant(ion_params['diffusivity'])
                ion_dict['De']      = Constant(ion_params['diffusivity'])                
                ion_dict['ki_init'] = Constant(ion_params['initial']['ics'])
                ion_dict['ke_init'] = Constant(ion_params['initial']['ecs'])                
                
                if 'source' in ion_params:
                    ion_dict['f_i'] = Constant(ion_params['source']['ics'])              
                    ion_dict['f_e'] = Constant(ion_params['source']['ecs'])                               
                else:
                    ion_dict['f_i'] = Constant(0.0)              
                    ion_dict['f_e'] = Constant(0.0)                               
            
                self.ion_list.append(ion_dict) 

            self.N_ions = len(self.ion_list) 
        else:
            if MPI.comm_world.rank == 0: print('Using default ionic species {Na, K, Cl}')


    def parse_tags(self, tags):

        allowed_tags = {'intra','extra','membrane','boundary'}

        tags_set = set(tags.keys())

        if MPI.comm_world.rank == 0: 

            # checks
            if not tags_set.issubset(allowed_tags):            
                print('ERROR: mismatch in tags!')
                print('Allowed tags:', allowed_tags)
                print('Input tags:', tags_set)              
                exit()

            # print info
            if isinstance(tags['intra'], collections.abc.Sequence):
                print("#Cell tags =", len(tags['intra']))           
            else:           
                print("Single cell tag")    

        if 'intra' in tags_set:
            self.intra_tags = tags['intra']
        else:
            if MPI.comm_world.rank == 0: print('ERROR: intra tag has to be provided!')
            exit()

        if 'extra' in tags_set:
            self.extra_tag = tags['extra']
        else:
            if MPI.comm_world.rank == 0: print('Setting default extra tag = 1')
            self.extra_tag = 1

        if 'membrane' in tags_set:
            self.gamma_tags = tags['membrane']    
        else:
            if MPI.comm_world.rank == 0: print('Setting default membrane tag = intra tag')
            self.gamma_tags = self.intra_tags

        if 'boundary' in tags_set:
            self.bound_tag  = tags['boundary']     
        else:
            if MPI.comm_world.rank == 0: print('Setting default boundary tag = 1')
            self.bound_tag = 1


        # trasform in tuple if needed        
        if isinstance(self.intra_tags, list): self.intra_tags = tuple(self.intra_tags)        
        if isinstance(self.gamma_tags, list): self.gamma_tags = tuple(self.gamma_tags)
        if isinstance(self.gamma_tags, int):  self.gamma_tags = (self.gamma_tags,)

    def init_ionic_model(self):    

        # init list
        ionic_tags = [] 

        # check all ICS tags are present in some ionic model
        for idx, model in enumerate(self.ionic_models):          
            ionic_tags.append(model.tags)

        ionic_tags = sorted(flatten_list(ionic_tags))
        gamma_tags = sorted(flatten_list([self.gamma_tags]))

        if ionic_tags != gamma_tags:
            print('ERROR: mismatch between gamma tags and ionic models tags.')
            print('Ionic model tags: ', ionic_tags)
            print('Membrane tags: '   , gamma_tags)
            exit()
                
        if MPI.comm_world.rank == 0:
            print("#Membrane tags =", len(gamma_tags))
            print("#Ionic models  =", len(self.ionic_models),'\n')
    

    def setup_domain(self):

        # rename files for readablity
        mesh_file             = self.input_files['mesh_file']     
        boundaries_file       = self.input_files['facets_file']
        intra_restriction_dir = self.input_files['intra_restriction_dir']
        extra_restriction_dir = self.input_files['extra_restriction_dir']           

        if mesh_file.endswith('.xml'): # TODO remove this

            if MPI.comm_world.rank == 0: print('Loading xml files...')          

            subdomains_file = self.input_files['subdomais_file']

            # load xml files
            self.mesh       = Mesh(mesh_file)
            self.subdomains = MeshFunction("size_t", self.mesh, subdomains_file)
            self.boundaries = MeshFunction("size_t", self.mesh, boundaries_file)

            # scale
            self.mesh.coordinates()[:] *= self.m_conversion_factor
                                    
            # Restrictions
            if MPI.comm_world.rank == 0: print('Creating mesh restrictions...')             
            self.interior = MeshRestriction(self.mesh, intra_restriction_dir)
            self.exterior = MeshRestriction(self.mesh, extra_restriction_dir)          

        elif mesh_file.endswith('.xdmf'):
        
            if MPI.comm_world.rank == 0: print('Loading xdmf files...')             

            self.mesh = Mesh()

            with XDMFFile(mesh_file) as f:
                f.read(self.mesh)               
                self.subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim(), 0) 
                f.read(self.subdomains)            
            
            with XDMFFile(boundaries_file) as f:
                self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1, 0)
                f.read(self.boundaries)

            # scale
            self.mesh.coordinates()[:] *= self.m_conversion_factor
                                           
            # Restrictions
            if MPI.comm_world.rank == 0: print('Creating mesh restrictions...')         
            self.interior = MeshRestriction(self.mesh, intra_restriction_dir)
            self.exterior = MeshRestriction(self.mesh, extra_restriction_dir)      

        else:

            if MPI.comm_world.rank == 0: print('ERROR: input format not supported.')    
            exit()      
    

    @abstractmethod
    def init(self):
        # Abstract method that must be implemented by concrete subclasses.
        pass
    
    @abstractmethod
    def setup_spaces(self):
        # Abstract method that must be implemented by concrete subclasses.
        pass

    @abstractmethod
    def setup_boundary_conditions(self):
        # Abstract method that must be implemented by concrete subclasses.
        pass


