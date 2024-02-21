import gmsh
import numpy.typing

import numpy   as np
import dolfinx as dfx

from mpi4py import MPI

def mark_subdomains_square(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTagsMetaClass:
    """ Function for marking subdomains of a unit square mesh with an interior square defined on [0.25, 0.75]^2.
    
    The subdomains have the following tags:
        - tag value 1 : inner square, (x, y) = [0.25, 0.75]^2
        - tag value 2 : outer square, (x, y) = [0, 1]^2 \ [0.25, 0.75]^2
    
    """ 
    def inside(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        """ Locator function for the inner square. """

        bool1 = np.logical_and(x[0] <= 0.75, x[0] >= 0.25) # True if inside inner box in x range
        bool2 = np.logical_and(x[1] <= 0.75, x[1] >= 0.25) # True if inside inner box in y range
        
        return np.logical_and(bool1, bool2)

    # Tag values
    INTRA = 1
    EXTRA = 2

    cell_dim = mesh.topology.dim
    
    # Generate mesh topology
    mesh.topology.create_entities(cell_dim)
    mesh.topology.create_connectivity(cell_dim, cell_dim - 1)
    
    # Get total number of cells and set default facet marker value to OUTER
    num_cells    = mesh.topology.index_map(cell_dim).size_local + mesh.topology.index_map(cell_dim).num_ghosts
    cell_marker  = np.full(num_cells, EXTRA, dtype = np.int32)

    # Get all facets
    inner_cells = dfx.mesh.locate_entities(mesh, cell_dim, inside)
    cell_marker[inner_cells] = INTRA

    cell_tags = dfx.mesh.meshtags(mesh, cell_dim, np.arange(num_cells, dtype = np.int32), cell_marker)

    return cell_tags

def mark_boundaries_square(mesh: dfx.mesh.Mesh) -> dfx.mesh.MeshTagsMetaClass:
    """ Function for marking boundaries of a unit square mesh with an interior square defined on [0.25, 0.75]^2
    
    The boundaries have the following tags:
        - tag value 3 : outer boundary (\partial\Omega) 
        - tag value 4 : interface gamma between inner and outer square
        - tag value 5 : interior facets

    """    
    def right(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        return np.logical_and(np.isclose(x[0], 0.75), y_range)

    def left(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        y_range = np.logical_and(x[1] >= 0.25, x[1] <= 0.75)
        return np.logical_and(np.isclose(x[0], 0.25), y_range)

    def bottom(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        return np.logical_and(np.isclose(x[1], 0.25), x_range)

    def top(x: numpy.typing.NDArray[np.float64]) -> numpy.typing.NDArray[np.bool_]:
        x_range = np.logical_and(x[0] >= 0.25, x[0] <= 0.75)
        return np.logical_and(np.isclose(x[1], 0.75), x_range)

    # Tag values
    PARTIAL_OMEGA = 3
    GAMMA         = 4
    DEFAULT       = 5

    facet_dim = mesh.topology.dim - 1

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)
    mesh.topology.create_connectivity(facet_dim, facet_dim + 1)

    # Get total number of facets
    num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts
    facet_marker = np.full(num_facets, DEFAULT, dtype = np.int32)

    # Get boundary facets
    bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    facet_marker[bdry_facets] = PARTIAL_OMEGA

    top_facets = dfx.mesh.locate_entities(mesh, facet_dim, top)
    facet_marker[top_facets] = GAMMA

    bottom_facets = dfx.mesh.locate_entities(mesh, facet_dim, bottom)
    facet_marker[bottom_facets] = GAMMA

    left_facets = dfx.mesh.locate_entities(mesh, facet_dim, left)
    facet_marker[left_facets] = GAMMA

    right_facets = dfx.mesh.locate_entities(mesh, facet_dim, right)
    facet_marker[right_facets] = GAMMA

    facet_tags = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)

    return facet_tags

def create_square_mesh_with_tags(N_cells: int,
                                 comm: MPI.Comm = MPI.COMM_WORLD,
                                 ghost_mode = dfx.mesh.GhostMode.shared_facet) \
        -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTagsMetaClass, dfx.mesh.MeshTagsMetaClass)):
    """ Create a square mesh of a square within a square, with the inner square defined on (x, y) = [0.25, 0.75]^2.

    Parameters
    ----------
    N_cells : int
        number of mesh cells in x and y direction
    
    comm : MPI.Comm
        MPI communicator, by default MPI.COMM_WORLD
        
    ghost_mode : dfx.mesh.GhostMode
        mode that specifies how ghost nodes are shared in parallel, by default dfx.mesh.GhostMode.shared_facet

    Returns
    -------
    mesh : dfx.mesh.Mesh
        dolfinx mesh

    subdomains : dfx.mesh.MeshTagsMetaClass
        subdomain mesh tags

    boundaries : dfx.mesh.MeshTagsMetaClass
        boundary facet tags
    """

    mesh = dfx.mesh.create_unit_square(comm, N_cells, N_cells,
                                    cell_type = dfx.mesh.CellType.triangle,
                                    ghost_mode = ghost_mode)

    subdomains  = mark_subdomains_square(mesh)
    boundaries  = mark_boundaries_square(mesh)

    return mesh, subdomains, boundaries


def create_circle_mesh_gmsh(comm: MPI.Comm = MPI.COMM_WORLD, partitioner = dfx.mesh.GhostMode.shared_facet):
    """Function for creating a circle mesh with a membrane dividing the circle into two subdomains.

    """
    # Geometry
    r = 3
    lcar = 1. / 4.


    gmsh.initialize()
    gmsh.model.add("mesh")

    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lcar)
    p1 = gmsh.model.geo.addPoint(0.0, +r, 0.0, lcar)
    p2 = gmsh.model.geo.addPoint(0.0, -r, 0.0, lcar)
    c0 = gmsh.model.geo.addCircleArc(p1, p0, p2)
    c1 = gmsh.model.geo.addCircleArc(p2, p0, p1)
    l0 = gmsh.model.geo.addLine(p2, p1)
    line_loop_left = gmsh.model.geo.addCurveLoop([c0, l0])
    line_loop_right = gmsh.model.geo.addCurveLoop([c1, -l0])
    semicircle_left = gmsh.model.geo.addPlaneSurface([line_loop_left])
    semicircle_right = gmsh.model.geo.addPlaneSurface([line_loop_right])

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [c0, c1], 1)
    gmsh.model.addPhysicalGroup(1, [l0], 2)
    gmsh.model.addPhysicalGroup(2, [semicircle_left], 1)
    gmsh.model.addPhysicalGroup(2, [semicircle_right], 2)
    gmsh.model.mesh.generate(2)

    #partitioner = dfx.mesh.create_cell_partitioner(dfx.mesh.GhostMode.shared_facet)
    mesh, subdomains, boundaries = dfx.io.gmshio.model_to_mesh(gmsh.model,
                                                            comm = comm,
                                                            rank = 0, 
                                                            gdim = 2, 
                                                            partitioner = partitioner)
    gmsh.finalize()

    return mesh, subdomains, boundaries