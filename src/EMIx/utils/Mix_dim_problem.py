from dolfin import *
from abc    import ABC, abstractmethod
from multiphenics   import *
import collections.abc
import time


def flatten_list(input_list):
    return [item for sublist in input_list for item in (sublist if isinstance(sublist, tuple) else [sublist])]


class Mixed_dimensional_problem(ABC):    

    t = Constant(0.0) 
        

    def __init__(self, input_files, tags, dt):      

        tic = time.perf_counter()       

        if MPI.comm_world.rank == 0: print("Reading input data:", input_files)

        # assign input arguments
        self.input_files = input_files              

        # parse tags
        self.parse_tags(tags)
        
        # init time step
        self.dt = Constant(dt)

        # in case some problem dependent init is needed
        self.init()
                        
        # setup FEM
        self.setup_domain()  
        self.setup_spaces() 
        self.setup_boundary_conditions()        

        # init empty ionic model (to be filled)
        self.ionic_models = []      

        if MPI.comm_world.rank == 0: print(f"Problem setup in {time.perf_counter() - tic:0.4f} seconds\n")   

    
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


        # trasform int in tuple if needed
        if isinstance(self.gamma_tags, int): self.gamma_tags = (self.gamma_tags,)
    

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
        mesh_file       = self.input_files['mesh_file']     
        boundaries_file = self.input_files['facets_file']
        intra_restriction_dir = self.input_files['intra_restriction_dir']
        extra_restriction_dir = self.input_files['extra_restriction_dir']           

        if mesh_file.endswith('.xml'): # TODO remove this

            if MPI.comm_world.rank == 0: print('Loading xml files...')          

            subdomains_file = self.input_files['subdomais_file']

            # load xml files
            self.mesh = Mesh(mesh_file)
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


