import typing
import matplotlib
import multiphenicsx.fem
import mpl_toolkits.axes_grid1

import numpy   as np
import dolfinx as dfx
import matplotlib.tri    as tri
import matplotlib.pyplot as plt

from mpi4py import MPI
from dfx_mesh_creation import create_square_mesh_with_tags

def plot_mesh(mesh: dfx.mesh.Mesh, ax: typing.Optional[plt.Axes] = None) -> plt.Axes:
    """Plot a mesh object on the provided (or, if None, the current) axes object."""

    if ax is None:
        ax = plt.gca()

    ax.set_aspect("equal")
    points = mesh.geometry.x
    cells = mesh.geometry.dofmap
    cells = reshape_adjacency_list(cells)
    tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(tria, color="k")

    return ax


def reshape_adjacency_list(adj_list: dfx.cpp.graph.AdjacencyList_int32):
    size = adj_list.num_nodes
    out = np.zeros((size, 3), dtype = np.int32)
    for i in range(size):
        out[i] = adj_list.links(i)
    
    return out

def plot_mesh_tags(mesh: dfx.mesh.Mesh, mesh_tags: dfx.mesh.MeshTagsMetaClass,
                   ax: typing.Optional[plt.Axes] = None, facet_tag_value: typing.Optional[int] = 0) -> plt.Axes:
    """Plot a mesh tags object on the provided (or, if None, the current) axes object."""

    if ax is None:
        ax = plt.gca()

    ax.set_aspect("equal")
    points = mesh.geometry.x
    colors = ["b", "r"]
    cmap = matplotlib.colors.ListedColormap(colors)
    cmap_bounds = [0, 0.5, 1]
    norm = matplotlib.colors.BoundaryNorm(cmap_bounds, cmap.N)
    assert mesh_tags.dim in (mesh.topology.dim, mesh.topology.dim - 1)

    if mesh_tags.dim == mesh.topology.dim:
        cells = mesh.geometry.dofmap
        cells = reshape_adjacency_list(cells)
        tria = tri.Triangulation(points[:, 0], points[:, 1], cells)
        cell_colors = np.zeros((cells.shape[0], ))
        cell_colors[mesh_tags.indices[mesh_tags.values == 1]] = 1
        mappable = ax.tripcolor(tria, cell_colors, edgecolor="k", cmap=cmap, norm=norm)

    elif mesh_tags.dim == mesh.topology.dim - 1:
        tdim = mesh.topology.dim
        cells_map = mesh.topology.index_map(mesh.topology.dim)
        num_cells = cells_map.size_local + cells_map.num_ghosts
        connectivity_cells_to_facets = mesh.topology.connectivity(tdim, tdim - 1)
        connectivity_cells_to_vertices = mesh.topology.connectivity(tdim, 0)
        connectivity_facets_to_vertices = mesh.topology.connectivity(tdim - 1, 0)
        dofmap = reshape_adjacency_list(mesh.geometry.dofmap)
        vertex_map = {topology_index: geometry_index
                      for c in range(num_cells)
                      for (topology_index, geometry_index) in zip(
                          connectivity_cells_to_vertices.links(c), dofmap[c])}
        linestyles = ["solid", "solid"] # Change first entry to (0, (5, 10)) for dashed lines
        lines = list()
        lines_colors_as_int = list()
        lines_colors_as_str = list()
        lines_linestyles = list()
        mesh_tags_to_color = mesh_tags.indices[mesh_tags.values == facet_tag_value]
        for c in range(num_cells):
            facets = connectivity_cells_to_facets.links(c)
            for f in facets:
                if f in mesh_tags_to_color:
                    value_f = 1
                else:
                    value_f = 0
                vertices = [vertex_map[v] for v in connectivity_facets_to_vertices.links(f)]
                lines.append(points[vertices][:, :2])
                lines_colors_as_int.append(value_f)
                lines_colors_as_str.append(colors[value_f])
                lines_linestyles.append(linestyles[value_f])
        mappable = matplotlib.collections.LineCollection(lines, cmap=cmap, norm=norm,
                                                         colors=lines_colors_as_str,
                                                         linestyles=lines_linestyles)
        mappable.set_array(np.array(lines_colors_as_int))
        ax.add_collection(mappable)
        ax.autoscale()

    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mappable, cax=cax, boundaries=cmap_bounds, ticks=cmap_bounds)

    return ax


def _plot_dofmap(coordinates: np.typing.NDArray[np.float64], ax: typing.Optional[plt.Axes] = None) -> plt.Axes:

    if ax is None:
        ax = plt.gca()
    
    text_offset = [1e-2, 1e-2]
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c="k", s=50)
    
    for c in np.unique(coordinates, axis=0):
        dofs_c = (coordinates == c).all(axis=1).nonzero()[0]
        text_c = np.array2string(dofs_c, separator=", ", max_line_width=10)
        ax.text(c[0] + text_offset[0], c[1] + text_offset[1], text_c, fontsize=15)
    
    return ax

def plot_dofmap(V: dfx.fem.FunctionSpace, ax: typing.Optional[plt.Axes] = None) -> plt.Axes:
    """Plot the DOFs in a function space object on the provied (or, if None, the current) axes object."""

    coordinates = V.tabulate_dof_coordinates().round(decimals=3)

    return _plot_dofmap(coordinates, ax)

def plot_dofmap_restriction(V: dfx.fem.FunctionSpace, restriction: multiphenicsx.fem.DofMapRestriction,
                            ax: typing.Optional[plt.Axes] = None) -> plt.Axes:
    """Plot the DOFs in a DofMapRestriction object on the provied (or, if None, the current) axes object."""

    coordinates = V.tabulate_dof_coordinates().round(decimals=3)
    
    return _plot_dofmap(coordinates[list(restriction.unrestricted_to_restricted.keys())], ax)


# Create mesh with boundaries and subdomains tagged
N = 4
mesh, subdomains, boundaries = create_square_mesh_with_tags(N_cells = N, comm = MPI.COMM_WORLD)
# mesh.topology.create_connectivity(mesh.topology.dim, 0)
# mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
# mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
ax = plot_mesh_tags(mesh, boundaries, facet_tag_value = 4)
# ax = plot_mesh_tags(mesh, subdomains)
P = 1
V1 = dfx.fem.FunctionSpace(mesh, ("Lagrange", P))
V2 = dfx.fem.FunctionSpace(mesh, ("Lagrange", P))
INTRA = 1
EXTRA = 2
cells_Omega1 = subdomains.indices[subdomains.values == INTRA]
cells_Omega2 = subdomains.indices[subdomains.values == EXTRA]

dofs_V1_Omega1 = dfx.fem.locate_dofs_topological(V1, subdomains.dim, cells_Omega1)
dofs_V2_Omega2 = dfx.fem.locate_dofs_topological(V2, subdomains.dim, cells_Omega2)

restriction_V1_Omega1 = multiphenicsx.fem.DofMapRestriction(V1.dofmap, dofs_V1_Omega1)
restriction_V2_Omega2 = multiphenicsx.fem.DofMapRestriction(V2.dofmap, dofs_V2_Omega2)

plot_dofmap_restriction(V2, restriction_V2_Omega2, ax)

ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.plot()
plt.show()