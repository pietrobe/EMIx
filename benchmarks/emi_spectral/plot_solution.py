import vtk
import pyvista as pv
import numpy as np
import argparse

ecs_id = 1

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Plot EMI solution')
    parser.add_argument('filedir', type=str, help='the directory with the solution files')
    args = parser.parse_args()
    filedir = args.filedir
    outfile = filedir + "/plot.png"
    cbar_title = r'$u$'
    reader = pv.get_reader(filedir + "/solution.xdmf")
    reader.set_active_time_value(0.01)
    sol = reader.read()
    subdomains = pv.read_meshio(filedir + "/subdomains.xdmf")
    sol["subdomains"] = subdomains["f"]

    phi_lim = (min(sol["phi_i"].min() , sol["phi_e"].min()),
            max(sol["phi_i"].max() , sol["phi_e"].max()))
    ecs = sol.extract_cells(sol["subdomains"] == ecs_id)
    ics = sol.extract_cells(sol["subdomains"] != ecs_id)
    ecsclip = ecs.clip(normal=(-1, -0.2, 0))
    icsclip = ics.clip(normal=(-1, -1, 0))

    scalar_bar_args=dict(title=cbar_title, vertical=False,
                                    height=0.1, width=0.6, position_x=0.2,
                                    position_y=0.00, title_font_size=70,
                                    label_font_size=40, font_family="times",
                                    fmt="%.1f")

    phi_abs_max = 0.4 #float(np.abs(phi_lim).max())
    pl = pv.Plotter(off_screen=True, window_size=(1300, 1100))
    pl.add_mesh(icsclip, clim=phi_abs_max, cmap="delta",
                #specular=1, specular_power=20,
                scalar_bar_args=scalar_bar_args)
    ecs_edges = ecsclip.extract_all_edges()
    pl.add_mesh(ecs_edges, clim=phi_abs_max, cmap="delta", show_scalar_bar=False, line_width=0.1)
    pl.camera_position = "yz"
    pl.camera.azimuth = 225
    pl.camera.elevation = 20
    pl.camera.zoom(1.3)
    pl.screenshot(outfile, transparent_background=False)


    # plot membrane
    outfile = filedir + "/mem_plot.png"
    membrane = ics.extract_surface().slice(normal=(-1, -1, 0)) #.extract_all_edges()
    ics_slice = ics.slice(normal=(-1, -1, 0))
    ecs_slice = ecs.slice(normal=(-1, -1, 0)).extract_all_edges()

    mem_scalar_bar = scalar_bar_args.copy()

    mem_scalar_bar["title"] = r'$v_{i0}$'
    #mem_scalar_bar["title_font_size"] = 46
    
    membrane["phi_mem"] = membrane["phi_i"] - membrane["phi_e"]
    phi_mem_abs_max = 0.4 #float(np.abs(membrane["phi_mem"]).max())
    pl = pv.Plotter(off_screen=True, window_size=(1300, 1100))
    pl.add_mesh(membrane, clim=phi_mem_abs_max, cmap="CMRmap",line_width=5,
                #specular=1, specular_power=20,
                scalar_bar_args=mem_scalar_bar)
    pl.add_mesh(ics_slice, clim=phi_mem_abs_max, cmap="delta",
             show_scalar_bar=False)
    pl.add_mesh(ecs_slice, clim=phi_mem_abs_max, cmap="delta",line_width=1,
             show_scalar_bar=False)
    pl.camera_position = "yz"
    pl.camera.azimuth = 225
    pl.camera.elevation = 20
    pl.camera.zoom(1.25)
    pl.screenshot(outfile, transparent_background=False)