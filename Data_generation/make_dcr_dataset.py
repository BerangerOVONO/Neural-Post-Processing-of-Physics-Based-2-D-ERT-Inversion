######################################################################################################################
# This is the building functions behind the geological subsurfaces used to efficiently train a neural network   ######
# with DC-current resistivity data in 2D without topography.                                                    ######  
#                                                                                                               ######
# Objective:                                                                                                    ######
# - Create uniform subsurface with different resistivity values                                                 ######
# - Introduce in the above subsurfaces different anomalies type and shape with different resistivity values     ######
# - Create multi-layers subsurface with different resistivity values                                            ######
# - Create multi-layers subsurface with faults                                                                  ######
# - Add anisotropy                                                                                              ######
# - simulate the resistivity contrast create by fluid injection                                                 ######
# - create a data base generator                                                                                ######
#                                                                                                               ######
# Author: Béranger OVONO EKORE                                                                                  ######
######################################################################################################################


# --- Preparation Libs ---
# Please make sure to install the following libraries before running this script.
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.meshtools import polytools as plc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygimli.physics import ert
from pygimli.viewer import showMesh
from pygimli.viewer.mpl import drawMesh
from scipy import interpolate
import random
import os
import shutil
#np.random.seed(1)

# --- Memory checking ---
import psutil
# - function to print the memory status during the data base generation -
def print_status():
    pg.info('###############################################################################################################')
    pg.info(dict(psutil.virtual_memory()._asdict()))
    pg.info('###############################################################################################################')

# --- parameters ---
# ! Please fill them up !
# data base name and location 
path = "/gpfs/projets/deepres/Data"
database_name = "/preInv_DCR_T0_WSnDD_saphir"
database_path = path + database_name
# csv file name for the data base index - will be used to load the data base for training -
csv_file = "index_multiLayers_part1.csv" 
#csv_file = "index_embedAnomaly_size6_part1.csv"
# - create the data base file -
if os.path.exists(database_path) == False:
    os.mkdir(database_path)
os.chdir(database_path)

# data base parameters settings
# The subsurface dimensions  and measurement array
# geological dimension 
left = 50
right = 50
depth = 25
paraBoundary=2.5 # the distance between the first and last electrode to the left and right boundaries
paraDX=0.5 # mesh fitness size between two electrodes
# - Measurement array parameters -
nb_elec = 96 # number of electrodes in the array
# -- configuration and file's name --
# Dipole-dipole is 'dd' and Wenner-Schlumbeerger is 'slm'
config_names = ['slm', 'dd'] 
# if you have your own sequences please put them in the directory=path and fill the bellow list with their names
config_files_names = ["WSR_96_AI_complete.bert", "DD_96_AI_complete.bert"]
if config_files_names != []:
    config_files = [os.path.join(path, file_name) for file_name in config_files_names]
else:
    config_files = None
# different noise levels considered 
noise_levels = [2] # in percentage, 2 mean 2% of absolute noise level and the relative data error is at 5e-5 by default
# anomalies resistivities
anomaly_res = [10, 100, 500, 1000, 2000] # this is the defaut but can be set in the file generate_embedAnomalies_dataset.py
# Grid mesh dimension on it we will sample the image over the unstructured mesh
nz = 101 # number of vertical cells (pixels) in the grid (the image becoming)
nx = 401 #4*nz # number of horizontal cell (pixels) in the grid (the image becoming)
x, z = np.linspace(-left, right, nx), np.linspace(-depth, 0, nz) # the gris extend
## The used grid (you may run this cell only one time)
grid = pg.createGrid(x, z, marker=1)
## Control points for the interfaces to be built
# Control points
nb_cp = 70 # number of control points used for the interface between two layers
control_points = np.linspace(-left, right, nb_cp).reshape(nb_cp,1)
# Norms for inversion objective function
# False is for L2 norm (the default one) and True is for L1 norme
robustData=False # The data misfit erroe
robustModel=False # The referential model(subsurface) penality
# saving the inversion quality rsme for the using régularisation
preInv_rmses = []
# Please indicate the name of the inversion quality saving file
#preInv_rmses_file_name = "preInv_rmse_misfitL2_reguL2_WSnDD_EA_size6_part1.txt"
preInv_rmses_file_name = "preInv_rmse_misfitL2_reguL2_WSnDD_ML_part1.txt" 

# --- Electrodes configuration ---
# The type of array used is located in the variable "scheme" and the function "add_scheme_node" add nodes at  
# the of position the electrodes from the array to the subsurface PLC.
# - create the electrical measurement array -
# By indicating the number of electrodes you would like to have in the array
def make_scheme_nb_elec(config='dd', nb_elec=96, paraBoundary=paraBoundary):
    scheme = ert.createData(elecs=np.linspace(start=-left+paraBoundary, stop=right-paraBoundary, num=nb_elec), schemeName=config)
    return scheme 
# By indicating the nominal electrode spacing 
def make_scheme_elec_spacing(config='dd', elec_spacing=10, paraBoundary=2.5):
    scheme = ert.createData(elecs=np.linspace(start=-left+paraBoundary, stop=right-paraBoundary
                                              , num=((left + right)// elec_spacing)+1), schemeName=config)
    return scheme 

# We will be using 96 electrodes Iris compatible Syscal multichannels commutative measurement system 
# Now let add the coordinates of each four-points array in the scheme 
def setSchemeSensorCoords(scheme, verbose=True):
    """ Set quadripole coordinates """
    nb_data = len(scheme['a'])
    zeros = list(np.zeros(nb_data))
    scheme['xa'] = zeros
    scheme['xb'] = zeros
    scheme['xm'] = zeros
    scheme['xn'] = zeros
    scheme['ya'] = zeros
    scheme['yb'] = zeros
    scheme['ym'] = zeros
    scheme['yn'] = zeros
    scheme['za'] = zeros
    scheme['zb'] = zeros
    scheme['zm'] = zeros
    scheme['zn'] = zeros
    for index in range(nb_data):
        for elec in "abmn":
            elec_id = int(scheme(elec)[index])
            elec_pos = scheme.sensorPosition(elec_id)
            scheme["x" + elec][index] = elec_pos.x()
            scheme["y" + elec][index] = elec_pos.y()
            scheme["z" + elec][index] = elec_pos.z()
    if verbose:
        print(scheme.tokenList())
        print(scheme)
# compute the pseudo investigation depth
def setSchemeInvestigationPoint(scheme, fourPointArayTypr='dd', verbose=True):
    """ Set depth of investion approximate coordinates """
    if fourPointArayTypr == 'dd':
        z_invest=4
    if fourPointArayTypr == 'slm':
        z_invest=6   
    nominalArray = np.array([scheme['a'], scheme['b'], scheme['m'], scheme['n']]).T 
    dx = scheme.sensors()[1].x() - scheme.sensors()[0].x()
    dy = scheme.sensors()[1].y() - scheme.sensors()[0].y()
    dz = scheme.sensors()[1].z() - scheme.sensors()[0].z()
    distRefElecNominalArray = np.max(nominalArray, axis=1) - np.min(nominalArray, axis=1)
    scheme['xreport'] = list(np.round(np.mean(np.array([scheme['xa'], scheme['xb'], scheme['xm'], scheme['xn']]).T, axis=1),3))
    scheme['yreport'] = list(np.round(np.mean(np.array([scheme['ya'], scheme['yb'], scheme['ym'], scheme['yn']]).T, axis=1), 3))
    scheme['zreport'] = list(np.round(-(((dx*distRefElecNominalArray)**2 + (dy*distRefElecNominalArray)**2 + (dz*distRefElecNominalArray)**2)**0.5)/z_invest, 3))
    if verbose:
        print(scheme.tokenList())
        print(scheme)
    return scheme
# set configurations for mmeasurement by whether loading theme (for or from field data) or by creating them using the two upper function 
# with the default pygimli sequences
def set_config(config_files=None, config_names=['slm'], nb_elec=96):
    schemes_list = []
    if config_files:
        for i, config_file in enumerate(config_files):
            scheme = ert.load(config_file)
            scheme['k'] = ert.createGeometricFactors(scheme, numerical=True)
            setSchemeSensorCoords(scheme, verbose=True)
            scheme = setSchemeInvestigationPoint(scheme, fourPointArayTypr=config_names[i])
            print(scheme)
            schemes_list.append(scheme)
            pg.info("scheme for config: " + config_names[i] + " has been set.")
    else:
        for config_name in config_names:
            scheme = make_scheme_nb_elec(config=config_name, nb_elec=nb_elec)
            scheme['k'] = ert.createGeometricFactors(scheme, numerical=True)
            setSchemeSensorCoords(scheme, verbose=True)
            scheme = setSchemeInvestigationPoint(scheme, fourPointArayTypr=config_name)
            schemes_list.append(scheme)
            pg.info("scheme for config: " + config_name + " has been set.")
    return schemes_list
# function to add nodes for electrodes in the subsurface PLC
def add_scheme_nodes(geom, scheme):
    for p in scheme.sensors():
        geom.createNode(p)
        geom.createNode(p - [0, 0.1])

# for an available scheme predefined with another sequencer for field data
# you will need to relocate the sensors to the PLC used for subsurfaces generation    
def relocate_scheme_sensors(scheme_file, xyz, save_bert_name=None):
    scheme = ert.load(scheme_file)
    scheme.setSensorPositions(xyz)
    scheme['k'] = ert.createGeometricFactors(scheme, numerical=True)
    if save_bert_name:
        scheme.save(save_bert_name+'.bert')
        pg.info('Save scheme to', save_bert_name+'.bert')
    print(scheme)
    return scheme
    
# --- The simulation and inversion functions ---
# convert pg data to pandas data frame
def gimli2pandas(data):
    """ pg to pd data frame"""
    data_geom_info = pd.DataFrame({'a' : data['a'], 'b' : data['b'], 'm' : data['m'], 'n' : data['n'],
                                   'k' : data['k'], 'u' : data['u'], 'rhoa' : data['rhoa'], 'err' : data['err'],
                                   'valid' : data['valid'], 'xa' : data['xa'], 'xb' : data['xb'], 'xm' : data['xm'], 'xn' : data['xn'],
                                   'ya' : data['ya'], 'yb' : data['yb'], 'ym' : data['ym'], 'yn' : data['yn'], 'xreport' : data['xreport'],
                                   'yreport' : data['yreport'], 'zreport' : data['zreport'], 'topo_effect' : data['topo_effect']})
    return data_geom_info
# simulate function for a giving mesh and scheme will also simulate the topographique effect
def simulate_data(mesh, scheme, resistivity, I=1, noiseLevel=None, noiseAbs=5e-5, seed=1, simulate_topo_effect=True, rand_noise_var=1):
    """ Simulate direct problem """
    if not noiseLevel:
        data = ert.simulate(mesh, scheme=scheme, res=resistivity)
    else:
        data = ert.simulate(mesh, scheme=scheme, res=resistivity, noiseLevel=noiseLevel, noiseAbs=noiseAbs, seed=seed)
    if simulate_topo_effect:
        K0 = ert.createGeometricFactors(scheme, numerical=False)
        K = ert.createGeometricFactors(scheme, numerical=True)
        topo_effect = K/K0
        print('Befor topo effect :', data['rhoa'])
        data['rhoa'] = data['rhoa'] + topo_effect
        print('After topo effect :', data['rhoa'])
        data['rand_noise'] = rand_noise_var * np.random.rand(len(data['rhoa']))
        data['rhoa'] = data['rhoa'] + data['rand_noise']
        data['topo_effect'] = topo_effect
        print('After topo effect and random noise :', data['rhoa'])
    data.remove(data['rhoa'] < 0)
    data.remove(data['valid'] < 1)
    data['u'] = I*(data['rhoa'] / data['k'])
    data_geom_info = gimli2pandas(data)
    return (data, data_geom_info)

# --- inversion function ---
# with the mesh paradomain of pygimli 
def inversion_pg(data, paraDepth=25, quality=33.6, paraMaxCellSize=50, paraDX=paraDX, paraBoundary=paraBoundary, limits=None, zWeight=None, lam=20, robustData=robustData, robustModel=robustModel):
    mgr = ert.ERTManager(data, verbose=False)
    mgr.invert(paraDepth=paraDepth, limits=limits, quality=quality, paraMaxCellSize=paraMaxCellSize, paraDX=paraDX, zWeight=zWeight, lam=lam
               , robustData=robustData, paraBoundary=paraBoundary, robustModel=robustModel)
    return mgr
# with a given mesh (may be a predesigned one for the inversion)
def inversion_pg_inMesh(data, inMesh, limits=None, zWeight=None, lam=20, robustData=robustData, robustModel=robustModel):
    mgr = ert.ERTManager(data, mesh=inMesh, verbose=True, limits=None, zWeight=None, lam=20)
    mgr.setMesh(inMesh)
    mgr.inver(limits=limits, zWeight=zWeight, lam=lam, robustData=robustData, robustModel=robustModel)
    mgr.showResult()
    plt.show()
    return mgr

# --- Gelogical subsurfaces ---
# ## Predifined anomalies type and shape
# Function to add the anomaly to a subsurface plc or mesh
def add_to_plc_or_mesh(world, poly, center, to_mesh=False, marker=None):
    if not to_mesh: # merge PLCs
        mt.polytools.setPolyRegionMarker(poly, marker=marker)
        geom = mt.mergePLC([world, poly], 0.001)
        geom.addRegionMarker((center[0], center[1]), marker=marker)
    else: # merge Meshes
        mesh_ano = mt.createMesh(poly, smooth=True)
        mesh_ano.setCellMarkers([marker]*mesh_ano.cellCount())
        geom = mt.mergeMeshes([world, mesh_ano])#, verbose=True)
        geom.addRegionMarker((center[0], center[1]), marker=marker)
    return geom
# Function to reaffect the right marker to the mesh ellements
def fix_PLCmarker_to_mesh(mesh, poly, marker, area=1):
    mesh_poly = mt.createMesh(poly, area=area, smooth=True)
    init_markers = mesh.cellMarkers()
    for pos in mesh_poly.cellCenters():
        cell = mesh.findCell(pos)
        if cell:
            init_markers[cell.id()] = marker
    mesh.setCellMarkers(init_markers) 
    return mesh
## Rectangular anomaly
def add_rectangle_anomaly(world, pos, size, isHole, marker=None, area=100, nodes=10, to_mesh=False):
    poly = mt.createRectangle(pos=pos, size=size, isHole=isHole, marker=marker, addNode=nodes, area=area)
    geom = add_to_plc_or_mesh(world, poly, center=pos, to_mesh=to_mesh, marker=marker)
    return geom, poly
## Cicle anomaly
def add_cicle_anomaly(world, pos, radius, isHole, marker=None, area=100, nSegments=10, to_mesh=False):
    poly = mt.createCircle(pos=pos, radius=radius, isClosed=True, isHole=isHole, area=area, nSegments=nSegments, marker=marker)
    geom = add_to_plc_or_mesh(world, poly, center=pos, to_mesh=to_mesh, marker=marker)
    return geom, poly   
## triangle anomaly
def add_triangle_anomaly(world, pos, side, isHole, marker=None, nodes=10, area=100, to_mesh=False):
    poly = mt.createPolygon([[pos[0]-side/2, pos[1]-side/2],
                                     [pos[0], pos[1]+side/2],
                                     [pos[0]+side/2, pos[1]-side/2]],
                                     isClosed=True, isHole=isHole, addNode=nodes, area=area, marker=marker)
    geom = add_to_plc_or_mesh(world, poly, center=pos, to_mesh=to_mesh, marker=marker)
    return geom, poly
## Right side moon anomaly
def add_moon_r_anomaly(world, pos, max_radius, isHole, marker=None, nodes=10, area=100, to_mesh=False):
    poly = mt.createPolygon([[pos[0]-max_radius/2, pos[1]+max_radius/2],
                                     [pos[0]-max_radius/2, pos[1]-max_radius/2],
                                     [pos[0]+max_radius, pos[1]-max_radius/2], pos],
                                     isClosed=True, addNodes= nodes, interpolate="spline", area=area, isHole=isHole, marker = marker)
    geom = add_to_plc_or_mesh(world, poly, center=pos, to_mesh=to_mesh, marker=marker)
    return geom, poly
## left side moon anomaly
def add_moon_l_anomaly(world, pos, max_radius, isHole, marker=None, nodes=10, area=100, to_mesh=False):
    poly = mt.createPolygon([[pos[0]+max_radius/2, pos[1]+max_radius/2],
                                     [pos[0]+max_radius/2, pos[1]-max_radius/2],
                                     [pos[0]-max_radius, pos[1]-max_radius/2], pos],
                                     isClosed=True, addNodes= nodes, interpolate="spline", area=area, isHole=isHole, marker = marker)
    geom = add_to_plc_or_mesh(world, poly, center=pos, to_mesh=to_mesh, marker=marker)
    return geom, poly
## block anomaly
def add_block_anomaly(world, pos, side, isHole, marker=None, nodes=10, area=100, to_mesh=False):
    poly = mt.createPolygon([[pos[0]-(3*side)/2, pos[1]+side/2],
                                       [pos[0]-side/2, pos[1]+side/2],
                                       [pos[0]-side/2, pos[1]+ (3*side)/2],
                                       [pos[0]+side/2, pos[1]+ (3*side)/2],
                                       [pos[0]+side/2, pos[1]+side/2],
                                       [pos[0]+(3*side)/2, pos[1]+side/2],
                                       [pos[0]+(3*side)/2, pos[1]-side/2],
                                       [pos[0]-(3*side)/2, pos[1]-side/2]],
                                     isClosed=True, isHole=isHole, addNode=nodes, area=area, marker=marker)
    geom = add_to_plc_or_mesh(world, poly, center=pos, to_mesh=to_mesh, marker=marker)
    return geom, poly
## Cup_like (bird_like) anomaly
def add_cup_anomaly(world, pos, side, isHole, marker=None, nodes=0, area=100, to_mesh=False):
    poly = mt.createPolygon([[pos[0]-side, pos[1]-side],
                                     [pos[0]-side/2, pos[1]-side/2],
                                     [pos[0]-side, pos[1]],
                                     [pos[0]-side, pos[1]+side],
                                      [pos[0]-side/2, pos[1]+side/2],
                                      [pos[0]+side/2, pos[1]+side/2],
                                      [pos[0]+side, pos[1]+side],
                                      [pos[0]+side, pos[1]],
                                      [pos[0]+side/2, pos[1]-side/2],
                                      [pos[0]+side, pos[1]-side]],
                                     isClosed=True, addNodes= nodes, interpolate="spline", area=area, isHole=isHole, marker = marker)
    geom = add_to_plc_or_mesh(world, poly, center=pos, to_mesh=to_mesh, marker=marker)
    return geom, poly
## Patatoid anomaly
def add_patatoid(world, num_points=20, radius=[1.0, 1.0], noise_var=0.2, center=[0, -5], marker=3, to_mesh=False, nodes=100, area=50):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius_var = noise_var* np.random.normal(size=num_points)
    x = center[0] + (radius[0] + radius_var) * np.cos(angles)
    y = center[1] + (radius[1] + radius_var) * np.sin(angles)
    poly = pg.meshtools.createPolygon(list(zip(x, y)), isClosed=True, marker=marker, area=area, addNodes=nodes, interpolate='spline')
    geom = add_to_plc_or_mesh(world, poly, center=center, to_mesh=to_mesh, marker=marker)
    return geom, poly

def rotate_point(point, angle_deg, center):
    angle = np.deg2rad(angle_deg)
    x, y = point[0]-center[0], point[1]-center[1]
    rotate_point = [x * np.cos(angle) - y * np.sin(angle) + center[0], y * np.cos(angle) + x * np.sin(angle)+ center[1]]
    return rotate_point
## Right side moon anomaly
def add_bar_anomaly(world, pos, top_radius, bot_radius, angle, isHole, marker=None, nodes=10, area=100, to_mesh=False):
    poly = mt.createPolygon([rotate_point([pos[0]-top_radius, pos[1]+top_radius], angle, pos),
                                     rotate_point([pos[0]+top_radius, pos[1]+top_radius], angle, pos),
                                     rotate_point([pos[0]+top_radius, pos[1]-bot_radius], angle, pos), 
                                     rotate_point([pos[0]-top_radius, pos[1]-bot_radius], angle, pos),],
                                     isClosed=True, addNodes= nodes, area=area, isHole=isHole, marker = marker)
    geom = add_to_plc_or_mesh(world, poly, center=pos, to_mesh=to_mesh, marker=marker)
    return geom, poly
## selecting an anomaly to add it into a predifined subsurface 
def add_anomaly(world, pos, size, type='square', isHole=False, marker=1, area=None, to_mesh=True):
    if type=='square':
        geom = add_rectangle_anomaly(world, pos=pos, size=size, isHole=isHole, marker=marker, area=area)
    if type=='cicle':
        geom = add_cicle_anomaly(world, pos=pos, radius=[(size*3)/4, size/2], isHole=isHole, marker=marker, area=area, nSegments=20)
    if type=='moon':
        geom = add_moon_r_anomaly(world, pos=pos, max_radius=size, isHole=isHole, marker=marker, nodes=20, area=area)
    if type=='triangle':
        geom = add_triangle_anomaly(world, pos=pos, side=size, isHole=isHole, marker=marker, nodes=10, area=area)
    if type=='block':
        geom = add_block_anomaly(world, pos=pos, side=size/3, isHole=isHole, marker=marker, nodes=10, area=area)
    if type=='cup':
        geom = add_cup_anomaly(world, pos=pos, side=size/2, isHole=isHole, marker=marker, nodes=0, area=area)
    return geom
    
# --- Conversion of a tri-mesh to quad-mesh in ordre to save it as a numpy ndarray image ---
# function to convert a rhomap to a vector 
def rhomap_to_resVect(mesh, rhomap):
    """ resistivity rhomap to a vector """
    rhomap = np.array(rhomap)
    data = np.array(mesh.cellMarkers())
    for n_marker in range(len(rhomap)):
        data[data==(rhomap[n_marker][0])] = rhomap[n_marker][1]
    return data
# convert a grid mesh to a 2d numpy ndarray
def meshGrid_to_numpyArray(nx, nz, resistivity):
    """ grid mesh to 2d numpy array """
    grid_array = np.flipud(np.array(resistivity).reshape(nz-1, nx-1))
    return grid_array
## functions to interpolate (convert) a mesh to another mesh (The two bellow function are from pygimli usage examples)
# the nearest
def nearest_neighbor_interpolation(inmesh, indata, outmesh, nan=99999.9):
    """ Nearest neighbor interpolation. """
    outdata = []
    for pos in outmesh.cellCenters():
        cell = inmesh.findCell(pos)
        if cell:
            outdata.append(indata[cell.id()])
        else:
            outdata.append(nan)
    return outdata
# the linear
def linear_interpolation(inmesh, indata, outmesh):
    """ Linear interpolation using `pg.interpolate()` """
    outdata = pg.Vector()  # empty
    pg.interpolate(srcMesh=inmesh, inVec=indata,
                   destPos=outmesh.cellCenters(), outVec=outdata)
    # alternatively you can use the interpolation matrix
    #outdata = inmesh.interpolationMatrix(outmesh.cellCenters()) * \
              #pg.core.cellDataToPointData(inmesh, indata)
    return outdata

# --- Multi-layers sudsurface ---
# We create the interface dipping by pertubating an intial line interface with depth z by add at each control point with a cummulative sum of a random noise.
# Function to create a fault on a layer
def make_fault(layer_depths:np.ndarray, nb_cp:int, fault_shift:int, fault_range:list, offset_angle:int=None):
    """ create a fault on layers of the subsurface"""
    shift = np.zeros(nb_cp)
    range_offset = fault_range[:]
    if offset_angle:
       del range_offset[0:offset_angle] 
    shift[range_offset] = fault_shift
    layer_depths = layer_depths - shift
    return layer_depths
# ## two layers
# the building function
def make_two_layer(world, left, right, depth, control_points, mean_depth, nb_cp, rand_var=5, tol_lim=0.9, marker=1, area=10, bmarker=None, fault_shift=None, fault_range=None):
    """ Create a 2 layers subsuface with a random interface dipping """
    top_lim = - tol_lim
    bottom_lim = - depth + tol_lim
    np.random.seed(random.randint(0, 2**32-1))
    cp_depths = -mean_depth - np.random.normal(0, rand_var, nb_cp).cumsum()
    if fault_shift and fault_shift:
        cp_depths = make_fault(cp_depths, nb_cp, fault_shift, fault_range, offset_angle=None)
    cp_depths[(cp_depths > top_lim)] = top_lim
    cp_depths[(cp_depths < bottom_lim)] = bottom_lim
    bottom_pos = np.concatenate((control_points, cp_depths.reshape(nb_cp, 1)), axis=1).tolist()
    layer_pos = bottom_pos + [[right,0], [-left, 0]]
    layer = mt.createPolygon(layer_pos, isClosed=True, marker=marker, area=area, markerPosition=bottom_pos[0])
    geom = world + layer
    return geom

# ## Three layers
# The building function
def make_three_layers(world, left, right, depth, control_points, mean_depths, nb_cp, rand_var=5, tol_lim=0.9, markers=[1, 2], area=100, offset_angle=None, fault_shift=None, fault_range=None, same_shape=True):
    """ Create a 2 layers subsuface with a random interface dipping """
    top_lim = - tol_lim
    bottom_lim = - depth + tol_lim
    bottom_layer = None
    top_layer = None
    np.random.seed(random.randint(0, 2**32-1))
    if same_shape:
        # the facier deformer
        depth_deform = np.random.normal(0, rand_var, nb_cp).cumsum()
        ## top layer 
        top_layer_depths = -mean_depths[0] - depth_deform
        # if on set and make the fault
        if fault_shift and fault_shift:
            top_layer_depths = make_fault(top_layer_depths, nb_cp, fault_shift, fault_range, offset_angle=offset_angle)
        # check and set the tolerance limites to able the mesh feasibility
        top_layer_depths[(top_layer_depths > top_lim)] = top_lim
        top_layer_depths[(top_layer_depths < bottom_lim)] = bottom_lim + tol_lim
        # create the top layer
        top_layer_pos = [[-left, top_lim/2], [right,top_lim/2]] + np.concatenate((control_points, top_layer_depths.reshape(nb_cp, 1)), axis=1)[::-1].tolist()
        top_layer = mt.createPolygon(top_layer_pos, isClosed=True, marker=markers[0], area=area-1)
        ## bottom layer
        bottom_layer_depths = -mean_depths[1] - depth_deform
        # if on set and make the fault
        if fault_shift and fault_shift:
            bottom_layer_depths = make_fault(bottom_layer_depths, nb_cp, fault_shift, fault_range, offset_angle=None)
        # check and set the tolerance limites to able the mesh feasibility
        index_layer_1_2_lim = (bottom_layer_depths >= top_layer_depths)
        bottom_layer_depths[index_layer_1_2_lim] = top_layer_depths[index_layer_1_2_lim] - tol_lim
        bottom_layer_depths[(bottom_layer_depths < bottom_lim)] = bottom_lim
        # create the bottom layer
        bottom_layer_pos = np.concatenate((control_points, bottom_layer_depths.reshape(nb_cp, 1)), axis=1).tolist() + [[right,-depth+tol_lim/2], [-left, -depth+tol_lim/2]]
        bottom_layer = mt.createPolygon(bottom_layer_pos, isClosed=True, marker = markers[1], area=area-2)
    else:
        ## top layer 
        # the facier deformer for the top layer
        depth_deform = np.random.normal(0, rand_var, nb_cp).cumsum()
        # the layer depths
        top_layer_depths = -mean_depths[0] - depth_deform
        # check and set the tolerance limites to able the mesh feasibility
        top_layer_depths[(top_layer_depths > top_lim)] = top_lim
        top_layer_depths[(top_layer_depths < bottom_lim)] = bottom_lim + 2*tol_lim
        # create the top layer
        top_layer_pos = [[-left, top_lim/2], [right,top_lim/2]] + np.concatenate((control_points, top_layer_depths.reshape(nb_cp, 1)), axis=1)[::-1].tolist()
        top_layer = mt.createPolygon(top_layer_pos, isClosed=True, marker=markers[0], area=area-1)
        ## bottom layer
        # the facier deformer for the bottom layer
        depth_deform = np.random.normal(0, rand_var, nb_cp).cumsum()
        # the layer depths
        bottom_layer_depths = -mean_depths[1] - depth_deform
        # check and set the tolerance limites to able the mesh feasibility
        bottom_layer_depths[(bottom_layer_depths > top_lim)] = top_lim - tol_lim
        index_layer_1_2_lim = (bottom_layer_depths >= top_layer_depths)
        bottom_layer_depths[index_layer_1_2_lim] = top_layer_depths[index_layer_1_2_lim] - tol_lim
        bottom_layer_depths[(bottom_layer_depths < bottom_lim)] = bottom_lim
        # create the bottom layer
        bottom_layer_pos = np.concatenate((control_points, bottom_layer_depths.reshape(nb_cp, 1)), axis=1).tolist() + [[right,-depth+tol_lim/2], [-left, -depth+tol_lim/2]]
        bottom_layer = mt.createPolygon(bottom_layer_pos, isClosed=True, marker = markers[1], area=area-2)
    # merge the geological PLCs
    geom = top_layer + world + bottom_layer
    return geom

# ## Four layers
# The building function
def make_four_layers(world, left, right, depth, control_points, mean_depths, nb_cp, rand_var=5, tol_lim=0.9, markers=[1, 2, 3], area=100, offset_angle=None, fault_shift=None, fault_range=None, same_shape=True):
    top_lim = - tol_lim
    bottom_lim = - depth + tol_lim
    layer1 = None
    layer2 = None
    layer3 = None
    np.random.seed(random.randint(0, 2**32-1))
    if same_shape:
        # the limite deformer
        depth_deform = np.random.normal(0, rand_var, nb_cp).cumsum()
        ## layer 1
        layer1_depths = -mean_depths[0] - depth_deform
        # if on set and make the fault
        if fault_shift and fault_range:
            layer1_depths = make_fault(layer1_depths, nb_cp, fault_shift, fault_range, offset_angle=offset_angle)
        # check and set the tolerance limites to able the mesh feasibility
        layer1_depths[(layer1_depths > top_lim)] = top_lim
        layer1_depths[(layer1_depths < bottom_lim)] = bottom_lim + 2*tol_lim
        # create the top layer
        layer1_pos = [[-left, top_lim/2], [right,top_lim/2]] + np.concatenate((control_points, layer1_depths.reshape(nb_cp, 1)), axis=1)[::-1].tolist()
        layer1 = mt.createPolygon(layer1_pos, isClosed=True, marker=markers[0], area=area-1)
        ## layer 2
        # the bottom of layer2
        layer2_bottom_depths = -mean_depths[1] - depth_deform
        # if on set and make the fault
        if fault_shift and fault_shift:
            layer2_bottom_depths = make_fault(layer2_bottom_depths, nb_cp, fault_shift, fault_range, offset_angle=offset_angle)
        # check and set the tolerance limites to able the mesh feasibility
        index_layer_1_2_lim = (layer2_bottom_depths >= layer1_depths)
        layer2_bottom_depths[index_layer_1_2_lim] = layer1_depths[index_layer_1_2_lim] - 2*tol_lim
        layer2_bottom_depths[(layer2_bottom_depths < bottom_lim)] = bottom_lim + tol_lim
        layer2_top_depths = layer1_depths
        layer2_pos = np.concatenate((control_points, layer2_bottom_depths.reshape(nb_cp, 1)), axis=1).tolist() + np.concatenate((control_points, layer2_top_depths.reshape(nb_cp, 1)), axis=1)[::-1].tolist()
        layer2 = mt.createPolygon(layer2_pos, isClosed=True, marker = markers[1], area=area-2)
        ## layer 4
        layer4_top_depths = -mean_depths[2] - depth_deform
        # if on set and make the fault
        if fault_shift and fault_shift:
            layer4_top_depths = make_fault(layer4_top_depths, nb_cp, fault_shift, fault_range, offset_angle=offset_angle)
        # check and set the tolerance limites to able the mesh feasibility
        index_layer_4_2_lim = (layer4_top_depths >= layer2_bottom_depths)
        layer4_top_depths[index_layer_4_2_lim] = layer2_bottom_depths[index_layer_4_2_lim] - tol_lim
        layer4_top_depths[(layer4_top_depths < bottom_lim)] = bottom_lim
        layer4_pos = np.concatenate((control_points, layer4_top_depths.reshape(nb_cp, 1)), axis=1).tolist() + [[right,-depth+tol_lim/2], [-left, -depth+tol_lim/2]]
        layer4 = mt.createPolygon(layer4_pos, isClosed=True, marker = markers[2], area=area-3)
        # the geologie PLC
    else:
        ## layer 1
        # the layer 1 limite deformer
        depth_deform = np.random.normal(0, rand_var, nb_cp).cumsum()
        # the layer depths
        layer1_depths = -mean_depths[0] - depth_deform
        # check and set the tolerance limites to able the mesh feasibility
        layer1_depths[(layer1_depths > top_lim)] = top_lim
        layer1_depths[(layer1_depths < bottom_lim)] = bottom_lim + 2*tol_lim
        # create the top layer
        layer1_pos = np.concatenate((control_points, layer1_depths.reshape(nb_cp, 1)), axis=1).tolist() + [[right,top_lim/2], [-left, top_lim/2]] 
        layer1 = mt.createPolygon(layer1_pos, isClosed=True, marker=markers[0], area=area-1)
        ## layer 2
        # the bottom layer 2 limite deformer
        depth_deform = np.random.normal(0, rand_var, nb_cp).cumsum()
        # the bottom of layer2
        layer2_bottom_depths = -mean_depths[1] - depth_deform
        # check and set the tolerance limites to able the mesh feasibility
        layer2_bottom_depths[(layer2_bottom_depths > top_lim)] = top_lim - tol_lim
        index_layer_1_2_lim = (layer2_bottom_depths >= layer1_depths)
        layer2_bottom_depths[index_layer_1_2_lim] = layer1_depths[index_layer_1_2_lim] - tol_lim
        layer2_bottom_depths[(layer2_bottom_depths < bottom_lim)] = bottom_lim + tol_lim
        layer2_top_depths = layer1_depths
        layer2_pos = np.concatenate((control_points, layer2_bottom_depths.reshape(nb_cp, 1)), axis=1).tolist() + np.concatenate((control_points, layer2_top_depths.reshape(nb_cp, 1)), axis=1)[::-1].tolist()
        layer2 = mt.createPolygon(layer2_pos, isClosed=True, marker = markers[1], area=area-2)
        ## layer 4
        # the layer 1 limite deformer
        depth_deform = np.random.normal(0, rand_var, nb_cp).cumsum()
        # the layer 4 top depths 
        layer4_top_depths = -mean_depths[2] - depth_deform
        # check and set the tolerance limites to able the mesh feasibility
        layer4_top_depths[(layer4_top_depths > top_lim)] = top_lim - 2*tol_lim
        index_layer_4_2_lim = (layer4_top_depths >= layer2_bottom_depths)
        layer4_top_depths[index_layer_4_2_lim] = layer2_bottom_depths[index_layer_4_2_lim] - tol_lim
        layer4_top_depths[(layer4_top_depths < bottom_lim)] = bottom_lim
        layer4_pos = np.concatenate((control_points, layer4_top_depths.reshape(nb_cp, 1)), axis=1).tolist() + [[right,-depth+tol_lim/2], [-left, -depth+tol_lim/2]]
        layer4 = mt.createPolygon(layer4_pos, isClosed=True, marker = markers[2], area=area-3)
        # the geologie PLC
    geom = world + layer1 + layer2 + layer4
    return geom

# ## Add anisotropy
# function will create anysotropy at the layer scale for a given porosity
def add_anisotropy(mesh, N_cells, rhomap, rho_poro, porosity, interpolate=True):
    """ transform an isotropic uni(or mult) layers to an anisotropic one at the
    layer scale for a given porosity """
    rand = np.random.rand(N_cells)
    if len(rhomap)<2:
        resistivity = rhomap[0][1] * np.ones_like(rand)
        resistivity[(rand < porosity[0])] = rho_poro[0]
    else:
        #rhomap_init = np.array(rhomap)
        resistivity = np.ones_like(rand)
        for cell in mesh.cells():
            if rand[cell.id()] < porosity[cell.marker()]:
                resistivity[cell.id()] = rho_poro[cell.marker()]
            else:
                resistivity[cell.id()] = rhomap[cell.marker()][1] 
    if interpolate:
        #resistivity = pg.interpolate(mesh, resistivity, mesh.cellCenters())
        resistivity = linear_interpolation(mesh, resistivity, mesh)
    return resistivity

# ## Simulate resistivity contraste due to a fluide injection in the subsurface
# We consider that the formed plume follow a gaussian (or a mixture of gaussian for multiple injections points)
# 2d gaussian
def mix_gaussian(cells, mu_x, mu_y, sigma_x, sigma_y, cor_coef, weights):
    X, Y = cells[:,0].reshape(-1,1),  cells[:,1].reshape(-1,1)
    G = (1/(2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - cor_coef**2))) * np.exp(-1/(2 * (1 - cor_coef**2)) * (
        ((X - mu_x)**2 / sigma_x**2) +((Y - mu_y)**2 / sigma_y**2) 
        -(2 * cor_coef * (X - mu_x) * (Y - mu_y)) / (sigma_x * sigma_y)))
    G = (G.dot(weights)).reshape(-1)
    G = G / np.max(G)
    return G
# Simulate the injection of the gaussian mixture anomaly into the subsurface 
def inject_into_mesh(mesh, res, rhomap, inj_res=10, enclosed=None): # exemple enclosed = [True, True, True]
    if len(rhomap)<2:
        res = rhomap[0][1] - res * (rhomap[0][1] - inj_res)
    else:
        rhomap_inj = np.array(rhomap)
        if enclosed:
            rhomap_inj = rhomap_inj[enclosed]
        n = len(rhomap_inj)
        resistivity = rhomap_inj[:,1] - res.reshape(-1, 1) * (rhomap_inj[:,1] - inj_res)
        for cell in mesh.cells():
            res[cell.id()] = rhomap[cell.marker()][1]
            for j in range(n):
                if cell.marker() == rhomap_inj[j][0]:
                    res[cell.id()] = resistivity[cell.id(), j]
    return res 

# --- Data base generation ---
# saving frame liste
def set_data_saving_frames(config_names=None, noise_levels=None,  verbose=True):
    """ Differents frame in the data base """
    rho_file_names = []
    rho_file_mesh_names = []
    rhoa_datFile_names = [[] for _ in  config_names for n in noise_levels]
    rhoa_pandaFile_names = [[] for _ in config_names for n in noise_levels]
    rhoa_imgFile_names = [[] for _ in config_names for n in noise_levels]
    inv_rho_file_names = [[] for _ in config_names for n in noise_levels]
    if verbose:
        pg.info("Data base saving frames have been initiated")
    return rho_file_names, rho_file_mesh_names, rhoa_datFile_names, rhoa_pandaFile_names, inv_rho_file_names, rhoa_imgFile_names                                
rho_file_names, rho_file_mesh_names, rhoa_datFile_names, rhoa_pandaFile_names, inv_rho_file_names, rhoa_imgFile_names = set_data_saving_frames(config_names=config_names, noise_levels=noise_levels)

# saving the frame (The csv indixes file)
def save_db_indexes_file(csv_file, rho_file_names=None, config_names=None, noise_levels=None, with_inv=True, save_img=True, verbose=True):
    """ Save the database indexes file into a csv """
    df_index_data = pd.DataFrame({'True resistivity img': rho_file_names, 'True resistivity tri mesh': rho_file_mesh_names})
    for noisLevel in noise_levels:
        for i, config_name in enumerate(config_names):
            df_index_data_rhoa_pg = pd.DataFrame({'Rhoa pg file from '+config_name+' with '+str(noisLevel)+' % of noise': rhoa_datFile_names[i]})
            df_index_data_rhoa_infos = pd.DataFrame({'Rhoa infos from '+config_name+' with '+str(noisLevel)+' % of noise': rhoa_pandaFile_names[i]})
            if not with_inv:
                df_index_data = pd.concat([df_index_data, df_index_data_rhoa_pg, df_index_data_rhoa_infos], ignore_index=False, axis=1)
            else:
                df_index_inv_rho = pd.DataFrame({'Pg pre-inversion of rhoa from '+config_name+' with '+str(noisLevel)+' % of noise': inv_rho_file_names[i]})
                #df_index_data = pd.concat([df_index_data, df_index_data_rhoa_pg, df_index_data_rhoa_infos, df_index_inv_rho], ignore_index=False, axis=1)
                df_index_data = pd.concat([df_index_data, df_index_data_rhoa_pg, df_index_inv_rho], ignore_index=False, axis=1)
    df_index_data.to_csv(csv_file, index=False)
    if verbose:
        pg.info("The database csv files names has been saved.")

## saving the data rho, rhoa and the inverted rho
# rho
def save_rho_data(name, mesh, resistivity, grid, nx, nz, resIsRhomap=False, rho_file_names=None, rho_file_mesh_names=None, save_img=True, cmin=None, cmax=None, cmap=None, logScal=True):
    """ save the subsurface resistivity into numpy array csv and/or a img file """
    if resIsRhomap:
        resistivity = rhomap_to_resVect(mesh, resistivity)
    grid_res =  nearest_neighbor_interpolation(mesh, resistivity, grid)
    grid_array = meshGrid_to_numpyArray(nx, nz, grid_res)
    grid_array = grid_array.astype(np.float32)  # Ensure the array is of type float32 for saving
    np.savetxt(name + '.csv', grid_array, fmt='%.3f', delimiter=",")
    rho_file_names.append(name+'.csv')
    mesh.save(name+'.bms')
    rho_file_mesh_names.append(name+'.bms')
    #plt.imshow(grid_array, extent=[x.min(), x.max(), z.min(), z.max()], cmap="jet")
    #plt.show()
    #if save_img:
    #    fig , ax= showMesh(mesh, data=resistivity, logScale=True, label=pg.unit('res'), showBoundary=False, showMesh=False, colorbar=False, cMap=cmap)#, cMin=cmin, cMax=cmax);
    #    ax.remove()
    #    fig.set_axis_off()
    #    plt.savefig(name + '.png', bbox_inches='tight', pad_inches=0, transparent=False, dpi=300, facecolor='white')
    #    plt.ioff()
    #    plt.close()
    pg.info('rho model has been saved')
# rhoa
def save_rhoa_data(name, data, data_vals='rhoa', save_img=True, cmin=None, cmax=None, cmap=None):
    """ save the apparent resistivity simulated  into pygimli data .dat  and/or a img file """
    data.save(name+'.dat') 
    #if save_img:
    #    fig, ax = pg.physics.ert.showERTData(data, vals=data[data_vals], colorbar=False, cMap=cmap)#, cMin=cmin, cMax=cmax);
    #    ax.remove()
    #    fig.set_axis_off()
    #    plt.savefig(name + '.png', bbox_inches='tight', pad_inches=0, transparent=False, dpi=300, facecolor='white')
    #   plt.ioff()
    #   plt.close()
    pg.info(data_vals + ' has been simulated and saved')
# inverted rho
def save_rho_inverted(name, mgr, grid, nx, nz, save_img=True, cmin=None, cmax=None, cmap=None, logScal=True):
    """ save the geophysic inversion result subsurface resistivity into numpy array csv and/or a img file """
    grid_res = nearest_neighbor_interpolation(mgr.paraDomain, mgr.model, grid)
    grid_array = meshGrid_to_numpyArray(nx, nz, grid_res)
    grid_array = grid_array.astype(np.float32)  # Ensure the array is of type float32 for saving
    np.savetxt(name + '.csv', grid_array, fmt='%.3f', delimiter=",")
    #lt.imshow(grid_array, extent=[x.min(), x.max(), z.min(), z.max()], cmap="jet")
    #plt.show()
    preInv_rmses.append(mgr.inv.relrms())
    np.savetxt(preInv_rmses_file_name, preInv_rmses, fmt='%.5f', delimiter=",")
    #if save_img:
    #    fig , ax= showMesh(mgr.paraDomain, np.log(mgr.model), label= 'Invert '+pg.unit('res'), colorbar=False, cMap=cmap)#, cMin=cmin, cMax=cmax);
    #    ax.remove()
    #    fig.set_axis_off()
    #    plt.savefig(name + '.png', bbox_inches='tight', pad_inches=0, transparent=False, dpi=300, facecolor='white')
    #    plt.ioff()
    #    plt.close()
    #    fig, ax = plt.subplots()
    #    ax.imshow(grid_array, extent=[x.min(), x.max(), z.min(), z.max()], cmap="jet")
    #    plt.show();
    #    print('grid shape:', grid_array.shape)
    #    plt.ioff()
    #    plt.close()
    pg.info('Geophys invert rho model has been saved')
    del mgr

# --- Simulation and inversion ---
# chech the rhomap markers in mesh before simulating
def is_allRhoMapMarkes_in_mesh(mesh, rhomap):
    return len(np.unique(mesh.cellMarkers())) == len(rhomap)
     
# PLC or mesh direct probleme/inverse simulation and saving unit
def simulate_n_invert_unit(plc_o_mesh, grid, nx, nz, area, scheme, res_map, rho_name, config_name=None, mesh_on=False,noiseLevel=None, noiseAbs=5e-5, seed=None
                           , rhoa_datFile_names=None, rhoa_pandaFile_names=None, rhoa_imgFile_names=None, save_img=False, with_inv=True , inv_rho_file_names=None
                           , paraDepth=None, quality=34.6, paraMaxCellSize=None, limits=None, zWeight=None, cmin=None, cmax=None, cmap=None):
    """ Simulate the direct probleme and save the data for rho and rhoa """
    # setting mesh for simulation
    if mesh_on: # the plc is simple plc
        add_scheme_nodes(plc_o_mesh, scheme)
        mesh_ano = mt.createMesh(plc_o_mesh, area=area, smooth=True)
        save_rho_data(name=rho_name, mesh=mesh_ano, resistivity=res_map, grid=grid, nx=nx, nz=nz, resIsRhomap=True, save_img=True, cmap=cmap, cmin=cmin, cmax=cmin)
    else: # the plc is a meshed plc
        mesh_ano=plc_o_mesh
    # rhoa data part
    data, data_geom_info = simulate_data(mesh_ano, scheme=scheme, resistivity=res_map, I=1, noiseLevel=noiseLevel, noiseAbs=noiseAbs, seed=seed)
    rhoa_name = 'rhoa_config_' + config_name + '_on_' + rho_name + 'nl'+ str(noiseLevel)
    save_rhoa_data(name=rhoa_name, data=data, data_vals='rhoa', save_img=save_img, cmin=cmin, cmax=cmax, cmap=cmap)
    rhoa_datFile_names[(config_names.index(config_name)*(len(config_name)-1)+noise_levels.index(noiseLevel))].append(rhoa_name+'.dat')
    rhoa_pandaFile_names[(config_names.index(config_name)*(len(config_name)-1)+noise_levels.index(noiseLevel))].append(rhoa_name+'.csv')
    #data_geom_info.to_csv(rhoa_name+'.csv')
    # inversion part
    if with_inv:
        mgr = inversion_pg(data, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits, zWeight=zWeight)
        rho_invert_name = 'Invert_' + rhoa_name
        save_rho_inverted(rho_invert_name, mgr, grid, nx, nz, save_img=True, cmin=cmin, cmax=cmax, cmap=cmap)
        inv_rho_file_names[(config_names.index(config_name)*(len(config_name)-1)+noise_levels.index(noiseLevel))].append(rho_invert_name+'.csv')
    if save_img:
        rhoa_imgFile_names[(config_names.index(config_name)*(len(config_name)-1)+noise_levels.index(noiseLevel))].append(rhoa_name+'.png')

# PLC or mesh direct probleme/inverse simulation and saving for different noise levels for a single measurement array
def simulate_accros_noisLevel(noise_levels, plc_o_mesh, grid, nx, nz, area, scheme, res_map, rho_name, config_name=None, mesh_on=False,noiseLevel=None, noiseAbs=5e-5, seed=None
                              , rhoa_datFile_names=None, rhoa_pandaFile_names=None, rhoa_imgFile_names=None, save_img=False, with_inv=True 
                              , inv_rho_file_names=None, paraDepth=None, quality=34.6, paraMaxCellSize=None, limits=None, zWeight=None, cmin=None, cmax=None, cmap=None):    
    """ Simulate data for the same measurement array with the different noise levels on the same subsurface """
    for noiseLevel in noise_levels:
        simulate_n_invert_unit(plc_o_mesh, grid, nx, nz, area, scheme, res_map, rho_name, config_name=config_name, mesh_on=mesh_on, noiseLevel=noiseLevel, noiseAbs=noiseAbs, seed=seed
                               , rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=save_img, with_inv=with_inv
                               , inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits, zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)
# PLC or mesh direct probleme/inverse simulation and saving for different noise levels for different measurement arrays
def simulate_accros_configurations(config_names, noise_levels, plc_o_mesh, grid, nx, nz, area, schemes_list, res_map, rho_name, mesh_on=False, noiseAbs=5e-5, seed=None
                                   , rhoa_datFile_names=None, rhoa_pandaFile_names=None, rhoa_imgFile_names=None, save_img=False, with_inv=True , inv_rho_file_names=None
                                   , paraDepth=None, quality=34.6, paraMaxCellSize=None, limits=None, zWeight=None, cmin=None, cmax=None, cmap=None):    
    """ Simulate data for different measurement arrays with the different noise levels on the same subsurface """
    for simul, config_name in enumerate(config_names):
        simulate_accros_noisLevel(noise_levels, plc_o_mesh=plc_o_mesh, grid=grid, nx=nx, nz=nz, area=area, scheme=schemes_list[simul], res_map=res_map, rho_name=rho_name, config_name=config_name
                                  , mesh_on=mesh_on, noiseAbs=noiseAbs, seed=seed, rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names 
                                  , save_img=save_img, with_inv=with_inv, inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize
                                  , limits=limits, zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)

# ## Generate homogeneous media with embed anomalies
def create_data_embed_anomaly(left, right, depth, grid, nx, nz, config_files=None, config_names=['dd', 'slm'], nb_elecg=96, inf_res=10, sup_res=1000, area=5, anomaly_res=[10, 1000], anomaly_size=[30, 50], anomaly_pad=[10, 5]
                            , anomaly_stride=[100, 50], rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names, rhoa_datFile_names=rhoa_datFile_names
                            , rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=False, with_inv=True , inv_rho_file_names=inv_rho_file_names
                            , paraDepth=depth, quality=32, paraMaxCellSize=20, limits=None, zWeight=None, seed=None, cmap='jet', cmin=None, cmax=None, ratio_multi_anomaly=3, fix_marker_area=0.1 ):
    """ Generate the database on one or two layers subsurfaces with enbed anomalies """
    # elecs arrays
    schemes_list = set_config(config_files=config_files, config_names=config_names, nb_elec=nb_elec)
        
    world = mt.createWorld(start=[-left, 0], end=[right, -depth], marker=0, area=area)
    add_scheme_nodes(world, schemes_list[0])
    homo_res = anomaly_res[::-1]
    #cmin, cmax = inf_res, sup_res
    #cmin, cmax = None, None
    # anomaly type 
    anomaly_type = ['square', 'cicle', 'moon', 'triangle'] ##, 'block', 'cup']
    anomaly_type1 = anomaly_type ##+ ['bar']
    for res in homo_res:
        num_ano = 1
        d_iter=1
        #if res==10:
        #    anomaly_res_run=[1000, 500, 100]
        #else:
        #    anomaly_res_run=anomaly_res
        for ano_res in anomaly_res:
            if ano_res != res:
                for ano_size in anomaly_size:
                    lim_left = -left + anomaly_pad[0] + 2*ano_size//3
                    lim_right = right - anomaly_pad[0] - 2*ano_size//3
                    lim_top = - anomaly_pad[1]//2 - 2*ano_size//3
                    lim_bottom = - depth + anomaly_pad[1]*2 + 3*ano_size//4
                    for pos_h in range(lim_left, lim_right + 1, anomaly_stride[0]):
                        for pos_v in range(lim_top, lim_bottom - 1, -anomaly_stride[1]):
                            pos = [pos_h, pos_v]
                            res_map_one_ano = [[0, res], [1, ano_res]]
                            print(pos)

                            # seed a new random
                            np.random.seed(random.randint(0, 2**32-1))
                            
                            ### simulate and inverse by group 
                            ## group1: homogeneous subsurface with one embed anomaly
                            for type in anomaly_type:
                                geom, poly = add_anomaly(world, pos=pos, size=ano_size, type=type, isHole=False, marker=1, area=area)
                                mesh = mt.createMesh(geom, area=area, smooth=True)
                                rho_name = 'rho_homo_res'+str(round(res,2))+'_'+type+'_'+str(round(ano_res,2))+'_'+str(num_ano)+'_size_'+str(ano_size)+'_posh'+str(round(pos_h,1))+'_posv'+str(round(abs(pos_v),1))+'_r_'+str(d_iter)
                                if is_allRhoMapMarkes_in_mesh(mesh, res_map_one_ano):
                                    save_rho_data(name=rho_name, mesh=mesh, resistivity=res_map_one_ano, grid=grid, nx=nx, nz=nz, resIsRhomap=True, rho_file_names=rho_file_names
                                                  , rho_file_mesh_names=rho_file_mesh_names, save_img=True, cmap=cmap, cmin=cmin, cmax=cmin)
                                    simulate_accros_configurations(config_names, noise_levels, mesh, grid, nx, nz, area, schemes_list, res_map=res_map_one_ano, rho_name=rho_name
                                                                   , mesh_on=False, noiseAbs=1e-6, seed=seed, rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names
                                                                   , rhoa_imgFile_names=rhoa_imgFile_names, save_img=True, with_inv=True , inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth
                                                                   , quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits, zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)
                            # clear memory
                            del geom, mesh, poly
                            # check memory status
                            print_status()
                            # saving
                            save_db_indexes_file(csv_file=csv_file, rho_file_names=rho_file_names, config_names=config_names, noise_levels=noise_levels, with_inv=True, save_img=True, verbose=True)
                            
                            ## group2: homogeneous and two layered subsurfaces with two and three different anomalies
                            for ratio in range(ratio_multi_anomaly):

                                # seed a new random
                                np.random.seed(random.randint(0, 2**32-1))
                                ## homogeneous subsurface with two and three different anomalies
                                meshes = []
                                rhomaps = []
                                rho_names = []
                                # two anomalies
                                type1 = random.choice(anomaly_type1)
                                if type1 == 'bar':
                                    dip_bar_born = pos[1] - lim_bottom
                                    dip_bar = np.random.randint(anomaly_pad[1], int(dip_bar_born)+anomaly_pad[1]*2)
                                    if (depth+pos[1]-dip_bar)>=ano_size+max(anomaly_pad) and abs(pos[1]+lim_top)>=ano_size+max(anomaly_pad):
                                        angle_bar = np.random.randint(-30, 30)
                                    else:
                                        angle_bar = 0
                                    geom_2ano, poly = add_bar_anomaly(world, pos, ano_size//2.5, dip_bar, angle_bar, isHole=False, marker=1, nodes=10, area=area, to_mesh=False)
                                else:
                                    geom_2ano, poly = add_anomaly(world, pos=pos, size=ano_size, type=type1, isHole=False, marker=1, area=area)
                                type2 = random.choice(anomaly_type1)
                                pos2 = [random.randint(lim_left, lim_right), random.randint(lim_bottom, lim_top)]
                                if type2 == 'bar':
                                    dip_bar_born = pos2[1] - lim_bottom
                                    dip_bar = np.random.randint(anomaly_pad[1], int(dip_bar_born)+anomaly_pad[1]*2)
                                    if (depth+pos2[1]-dip_bar)>=ano_size+max(anomaly_pad) and abs(pos2[1]+lim_top)>=ano_size+max(anomaly_pad):
                                        angle_bar = np.random.randint(-30, 30)
                                    else:
                                        angle_bar = 0
                                    geom_2ano, poly = add_bar_anomaly(geom_2ano, pos2, ano_size//2.5, dip_bar, angle_bar, isHole=False, marker=2, nodes=10, area=area, to_mesh=False)
                                else:
                                    geom_2ano, poly = add_anomaly(geom_2ano, pos=pos2, size=ano_size, type=type2, isHole=False, marker=2, area=area)
                                mesh_2ano = mt.createMesh(geom_2ano, area=area, smooth=True)
                                res_map_two_ano = [[0, res], [1, ano_res], [2, random.randint(inf_res, sup_res)]]
                                # check if the mesh contains all the markers
                                if not is_allRhoMapMarkes_in_mesh(mesh_2ano, res_map_two_ano):
                                    mesh_2ano = fix_PLCmarker_to_mesh(mesh_2ano, poly, marker=2, area=fix_marker_area)
                                meshes.append(mesh_2ano)
                                rhomaps.append(res_map_two_ano)
                                rho_names.append('rho_homo_res' + str(round(res,2)) + '_two_anomalies_'+str(round(ano_res,2))+'_'+str(num_ano)+'_size_'+str(ano_size)+'_posh'+str(round(pos_h,1))+'_posv'+str(round(abs(pos_v),1))+'_r_'+str(d_iter))
                                # add a third anomaly
                                type3 = random.choice(anomaly_type)
                                pos3 = [random.randint(lim_left, lim_right), random.randint(lim_bottom, lim_top)]
                                geom_3ano, poly = add_anomaly(geom_2ano, pos=pos3, size=ano_size, type=type3, isHole=False, marker=3, area=area)
                                mesh_3ano = mt.createMesh(geom_3ano, area=area, smooth=True)
                                res_map_three_ano = [[0, res], [1, ano_res], [2, random.randint(inf_res, sup_res)], [3, random.randint(inf_res, sup_res)]]
                                # check if the mesh contains all the markers
                                if not is_allRhoMapMarkes_in_mesh(mesh_3ano, res_map_three_ano):
                                    mesh_3ano = fix_PLCmarker_to_mesh(mesh_3ano, poly, marker=3, area=fix_marker_area)
                                meshes.append(mesh_3ano)
                                rhomaps.append(res_map_three_ano)
                                rho_names.append('rho_homo_res'+str(round(res,2))+'_three_anomalies_'+str(round(ano_res,2))+'_'+str(num_ano)+'_size_'+str(ano_size)+'_posh'+str(round(pos_h,1))+'_posv'+str(round(abs(pos_v),1))+'_r_'+str(d_iter))    
                                ## two layered subsurfaces with one and two different anomalies
                                world_l2 = make_two_layer(world, left, right, depth, control_points, random.choice([depth/5, depth*2/5]), nb_cp, rand_var=random.choice([0.5, 1]), tol_lim=0.1, marker=1, area=area)
                                # with one embed anomaly
                                if type1 == 'bar':
                                    dip_bar_born = pos[1] - lim_bottom
                                    dip_bar = np.random.randint(anomaly_pad[1], int(dip_bar_born)+anomaly_pad[1]*2)
                                    if (depth+pos[1]-dip_bar)>=ano_size+max(anomaly_pad) and abs(pos[1]+lim_top)>=ano_size+max(anomaly_pad):
                                        angle_bar = np.random.randint(-30, 30)
                                    else:
                                        angle_bar = 0
                                    geom_l2, poly = add_bar_anomaly(world_l2, pos, ano_size//2.5, dip_bar, angle_bar, isHole=False, marker=2, nodes=10, area=area, to_mesh=False)
                                else:
                                    geom_l2, poly = add_anomaly(world_l2, pos=pos, size=ano_size, type=type1, isHole=False, marker=2, area=area)
                                res_l2_ano1 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)], [2, random.randint(inf_res, sup_res)]]
                                mesh_l2_ano1 = mt.createMesh(geom_l2, area=area, smooth=True)
                                # check if the mesh contains all the markers
                                mesh_l2_ano1 = fix_PLCmarker_to_mesh(mesh_l2_ano1, poly, marker=2, area=fix_marker_area)
                                rhomaps.append(res_l2_ano1)
                                meshes.append(mesh_l2_ano1)
                                rho_names.append('two_layer_top_' + str(round(res_l2_ano1[0][1],2))+'_bottom_' + str(round(res_l2_ano1[1][1],2))+'_1ano_'+str(num_ano)+'_size_'+str(ano_size)+'_posh'+str(round(pos_h,1))+'_posv'+str(round(abs(pos_v),1))+'_r_'+str(d_iter))
                                # with two embed anomalies
                                if type2 == 'bar':
                                    dip_bar_born = pos2[1] - lim_bottom
                                    dip_bar = np.random.randint(anomaly_pad[1], int(dip_bar_born)+anomaly_pad[1]*2)
                                    if (depth+pos2[1]-dip_bar)>=ano_size+max(anomaly_pad) and abs(pos2[1]+lim_top)>=ano_size+max(anomaly_pad):
                                        angle_bar = np.random.randint(-30, 30)
                                    else:
                                        angle_bar = 0
                                    geom_l2, poly = add_bar_anomaly(geom_l2, pos2, ano_size//2.5, dip_bar, angle_bar, isHole=False, marker=3, nodes=10, area=area, to_mesh=False)
                                else:
                                    geom_l2, poly = add_anomaly(geom_l2, pos=pos2, size=ano_size, type=type2, isHole=False, marker=3, area=area)  
                                res_l2_ano2 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)], [2, random.randint(inf_res, sup_res)], [3, random.randint(inf_res, sup_res)]]
                                rhomaps.append(res_l2_ano2) 
                                mesh_l2_ano2 = mt.createMesh(geom_l2, area=area, smooth=True)
                                # check if the mesh contains all the markers
                                mesh_l2_ano2 = fix_PLCmarker_to_mesh(mesh_l2_ano2, poly, marker=3, area=fix_marker_area)
                                meshes.append(mesh_l2_ano2)
                                rho_names.append('two_layer_top_' + str(round(res_l2_ano2[0][1],2))+'_bottom_' + str(round(res_l2_ano2[1][1],2))+'_2ano_'+str(num_ano)+'_size_'+str(ano_size)+'_posh'+str(round(pos_h,1))+'_posv'+str(round(abs(pos_v),1))+'_r_'+str(d_iter))
                                ## simulate and inverse
                                for k, mesh in enumerate(meshes):
                                    if is_allRhoMapMarkes_in_mesh(mesh, rhomaps[k]):
                                        save_rho_data(name=rho_names[k], mesh=mesh, resistivity=rhomaps[k], grid=grid, nx=nx, nz=nz, resIsRhomap=True, rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names
                                                    , save_img=True, cmap=cmap, cmin=cmin, cmax=cmin)
                                        simulate_accros_configurations(config_names, noise_levels, mesh, grid, nx, nz, area, schemes_list, res_map=rhomaps[k], rho_name=rho_names[k], mesh_on=False, noiseAbs=1e-6, seed=seed
                                                    , rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=True, with_inv=True 
                                                    , inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits
                                                    , zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)
                                
                                # clear memory
                                del meshes, rhomaps, rho_names, geom_2ano, mesh_2ano, geom_3ano, poly, mesh_3ano, world_l2, geom_l2, mesh_l2_ano1, mesh_l2_ano2
                                # check memory status
                                print_status()
                                # saving
                                save_db_indexes_file(csv_file=csv_file, rho_file_names=rho_file_names, config_names=config_names, noise_levels=noise_levels, with_inv=True, save_img=True, verbose=True)
                                d_iter+=1
        num_ano += 1        


# ## Generate homogeneous media with embed anomalies with different sizes
def create_data_embed_anomaly_multi_size(left, right, depth, grid, nx, nz, config_files=None, config_names=['dd', 'slm'], nb_elecg=96, inf_res=10, sup_res=1000, area=5, anomaly_res=[10, 1000], anomaly_size=[30, 50], anomaly_pad=[10, 5]
                            , anomaly_stride=[100, 50], rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names, rhoa_datFile_names=rhoa_datFile_names
                            , rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=False, with_inv=True , inv_rho_file_names=inv_rho_file_names
                            , paraDepth=depth, quality=32, paraMaxCellSize=20, limits=None, zWeight=None, seed=None, cmap='jet', cmin=None, cmax=None, ratio_multi_anomaly=3, fix_marker_area=0.1 ):
    """ Generate the database on one or two layers subsurfaces with enbed anomalies """
    # elecs arrays
    schemes_list = set_config(config_files=config_files, config_names=config_names, nb_elec=nb_elec)
        
    world = mt.createWorld(start=[-left, 0], end=[right, -depth], marker=0, area=area)
    add_scheme_nodes(world, schemes_list[0])
    homo_res = anomaly_res[::-1]
    #cmin, cmax = inf_res, sup_res
    #cmin, cmax = None, None
    # anomaly type 
    anomaly_type = ['square', 'cicle', 'moon', 'triangle'] ##, 'block', 'cup']
    anomaly_type1 = anomaly_type ##+ ['bar']
    for res in homo_res:
        num_ano = 1
        d_iter=1
        if res in [2000, 1000]:
            anomaly_res_run = []
        if res==500:
            anomaly_res_run = [10]
        if res in [100, 10]:
            anomaly_res_run = anomaly_res
        for ano_res in anomaly_res_run:
            if ano_res != res:
                for ano_size in anomaly_size:
                    lim_left = -left + anomaly_pad[0] + 2*ano_size//3
                    lim_right = right - anomaly_pad[0] - 2*ano_size//3
                    lim_top = - anomaly_pad[1]//2 - 2*ano_size//3
                    lim_bottom = - depth + anomaly_pad[1]*2 + 3*ano_size//4
                    for pos_h in range(lim_left, lim_right + 1, anomaly_stride[0]):
                        for pos_v in range(lim_top, lim_bottom - 1, -anomaly_stride[1]):
                            pos = [pos_h, pos_v]
                            res_map_one_ano = [[0, res], [1, ano_res]]
                            print(pos)

                            # seed a new random
                            np.random.seed(random.randint(0, 2**32-1))
                            
                            ## group: homogeneous and two layered subsurfaces with two and three different anomalies
                            for ratio in range(ratio_multi_anomaly):
                                # seed a new random
                                np.random.seed(random.randint(0, 2**32-1))
                                ## homogeneous subsurface with two and three different anomalies
                                meshes = []
                                rhomaps = []
                                rho_names = []
                                # two anomalies
                                type1 = random.choice(anomaly_type1)
                                if type1 == 'bar':
                                    dip_bar_born = pos[1] - lim_bottom
                                    dip_bar = np.random.randint(anomaly_pad[1], int(dip_bar_born)+anomaly_pad[1]*2)
                                    if (depth+pos[1]-dip_bar)>=ano_size+max(anomaly_pad) and abs(pos[1]+lim_top)>=ano_size+max(anomaly_pad):
                                        angle_bar = np.random.randint(-30, 30)
                                    else:
                                        angle_bar = 0
                                    geom_2ano, poly = add_bar_anomaly(world, pos, ano_size//2.5, dip_bar, angle_bar, isHole=False, marker=1, nodes=10, area=area, to_mesh=False)
                                else:
                                    geom_2ano, poly = add_anomaly(world, pos=pos, size=ano_size, type=type1, isHole=False, marker=1, area=area)
                                type2 = random.choice(anomaly_type1)
                                pos2 = [random.randint(lim_left, lim_right), random.randint(lim_bottom, lim_top)]
                                if type2 == 'bar':
                                    dip_bar_born = pos2[1] - lim_bottom
                                    dip_bar = np.random.randint(anomaly_pad[1], int(dip_bar_born)+anomaly_pad[1]*2)
                                    if (depth+pos2[1]-dip_bar)>=ano_size+max(anomaly_pad) and abs(pos2[1]+lim_top)>=ano_size+max(anomaly_pad):
                                        angle_bar = np.random.randint(-30, 30)
                                    else:
                                        angle_bar = 0
                                    geom_2ano, poly = add_bar_anomaly(geom_2ano, pos2, ano_size//2.5, dip_bar, angle_bar, isHole=False, marker=2, nodes=10, area=area, to_mesh=False)
                                else:
                                    ano_size2 = ano_size-1
                                    geom_2ano, poly = add_anomaly(geom_2ano, pos=pos2, size=ano_size2, type=type2, isHole=False, marker=2, area=area)
                                mesh_2ano = mt.createMesh(geom_2ano, area=area, smooth=True)
                                res_map_two_ano = [[0, res], [1, ano_res], [2, random.randint(inf_res, sup_res)]]
                                # check if the mesh contains all the markers
                                if not is_allRhoMapMarkes_in_mesh(mesh_2ano, res_map_two_ano):
                                    mesh_2ano = fix_PLCmarker_to_mesh(mesh_2ano, poly, marker=2, area=fix_marker_area)
                                meshes.append(mesh_2ano)
                                rhomaps.append(res_map_two_ano)
                                rho_names.append('rho_homo_res' + str(round(res,2)) + '_two_anomalies_'+str(round(ano_res,2))+'_'+str(num_ano)+'_multi_size_'+str(ano_size)+'_posh'+str(round(pos_h,1))+'_posv'+str(round(abs(pos_v),1))+'_r_'+str(d_iter))
                                # add a third anomaly
                                type3 = random.choice(anomaly_type)
                                pos3 = [random.randint(lim_left, lim_right), random.randint(lim_bottom, lim_top)]
                                ano_size3 = ano_size-2
                                geom_3ano, poly = add_anomaly(geom_2ano, pos=pos3, size=ano_size3, type=type3, isHole=False, marker=3, area=area)
                                mesh_3ano = mt.createMesh(geom_3ano, area=area, smooth=True)
                                res_map_three_ano = [[0, res], [1, ano_res], [2, random.randint(inf_res, sup_res)], [3, random.randint(inf_res, sup_res)]]
                                # check if the mesh contains all the markers
                                if not is_allRhoMapMarkes_in_mesh(mesh_3ano, res_map_three_ano):
                                    mesh_3ano = fix_PLCmarker_to_mesh(mesh_3ano, poly, marker=3, area=fix_marker_area)
                                meshes.append(mesh_3ano)
                                rhomaps.append(res_map_three_ano)
                                rho_names.append('rho_homo_res'+str(round(res,2))+'_three_anomalies_'+str(round(ano_res,2))+'_'+str(num_ano)+'_multi_size_'+str(ano_size)+'_posh'+str(round(pos_h,1))+'_posv'+str(round(abs(pos_v),1))+'_r_'+str(d_iter))    
                                ## two layered subsurfaces with one and two different anomalies
                                world_l2 = make_two_layer(world, left, right, depth, control_points, random.choice([depth/5, depth*2/5]), nb_cp, rand_var=random.choice([0.5, 1]), tol_lim=0.1, marker=1, area=area)
                                # with one embed anomaly
                                if type1 == 'bar':
                                    dip_bar_born = pos[1] - lim_bottom
                                    dip_bar = np.random.randint(anomaly_pad[1], int(dip_bar_born)+anomaly_pad[1]*2)
                                    if (depth+pos[1]-dip_bar)>=ano_size+max(anomaly_pad) and abs(pos[1]+lim_top)>=ano_size+max(anomaly_pad):
                                        angle_bar = np.random.randint(-30, 30)
                                    else:
                                        angle_bar = 0
                                    geom_l2, poly = add_bar_anomaly(world_l2, pos, ano_size//2.5, dip_bar, angle_bar, isHole=False, marker=2, nodes=10, area=area, to_mesh=False)
                                else:
                                    ano_size2 = ano_size-1
                                    geom_l2, poly = add_anomaly(world_l2, pos=pos, size=ano_size2, type=type1, isHole=False, marker=2, area=area)
                                res_l2_ano1 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)], [2, random.randint(inf_res, sup_res)]]
                                mesh_l2_ano1 = mt.createMesh(geom_l2, area=area, smooth=True)
                                # check if the mesh contains all the markers
                                mesh_l2_ano1 = fix_PLCmarker_to_mesh(mesh_l2_ano1, poly, marker=2, area=fix_marker_area)
                                rhomaps.append(res_l2_ano1)
                                meshes.append(mesh_l2_ano1)
                                rho_names.append('two_layer_top_' + str(round(res_l2_ano1[0][1],2))+'_bottom_' + str(round(res_l2_ano1[1][1],2))+'_1ano_'+str(num_ano)+'_multi_size_'+str(ano_size)+'_posh'+str(round(pos_h,1))+'_posv'+str(round(abs(pos_v),1))+'_r_'+str(d_iter))
                                # with two embed anomalies
                                if type2 == 'bar':
                                    dip_bar_born = pos2[1] - lim_bottom
                                    dip_bar = np.random.randint(anomaly_pad[1], int(dip_bar_born)+anomaly_pad[1]*2)
                                    if (depth+pos2[1]-dip_bar)>=ano_size+max(anomaly_pad) and abs(pos2[1]+lim_top)>=ano_size+max(anomaly_pad):
                                        angle_bar = np.random.randint(-30, 30)
                                    else:
                                        angle_bar = 0
                                    geom_l2, poly = add_bar_anomaly(geom_l2, pos2, ano_size//2.5, dip_bar, angle_bar, isHole=False, marker=3, nodes=10, area=area, to_mesh=False)
                                else:
                                    ano_size3 = ano_size-2
                                    geom_l2, poly = add_anomaly(geom_l2, pos=pos2, size=ano_size3, type=type2, isHole=False, marker=3, area=area)  
                                res_l2_ano2 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)], [2, random.randint(inf_res, sup_res)], [3, random.randint(inf_res, sup_res)]]
                                rhomaps.append(res_l2_ano2) 
                                mesh_l2_ano2 = mt.createMesh(geom_l2, area=area, smooth=True)
                                # check if the mesh contains all the markers
                                mesh_l2_ano2 = fix_PLCmarker_to_mesh(mesh_l2_ano2, poly, marker=3, area=fix_marker_area)
                                meshes.append(mesh_l2_ano2)
                                rho_names.append('two_layer_top_' + str(round(res_l2_ano2[0][1],2))+'_bottom_' + str(round(res_l2_ano2[1][1],2))+'_2ano_'+str(num_ano)+'_multi_size_'+str(ano_size)+'_posh'+str(round(pos_h,1))+'_posv'+str(round(abs(pos_v),1))+'_r_'+str(d_iter))
                                ## simulate and inverse
                                for k, mesh in enumerate(meshes):
                                    if is_allRhoMapMarkes_in_mesh(mesh, rhomaps[k]):
                                        save_rho_data(name=rho_names[k], mesh=mesh, resistivity=rhomaps[k], grid=grid, nx=nx, nz=nz, resIsRhomap=True, rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names
                                                    , save_img=True, cmap=cmap, cmin=cmin, cmax=cmin)
                                        simulate_accros_configurations(config_names, noise_levels, mesh, grid, nx, nz, area, schemes_list, res_map=rhomaps[k], rho_name=rho_names[k], mesh_on=False, noiseAbs=1e-6, seed=seed
                                                    , rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=True, with_inv=True 
                                                    , inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits
                                                    , zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)
                                
                                # clear memory
                                del meshes, rhomaps, rho_names, geom_2ano, mesh_2ano, geom_3ano, poly, mesh_3ano, world_l2, geom_l2, mesh_l2_ano1, mesh_l2_ano2
                                # check memory status
                                print_status()
                                # saving
                                save_db_indexes_file(csv_file=csv_file, rho_file_names=rho_file_names, config_names=config_names, noise_levels=noise_levels, with_inv=True, save_img=True, verbose=True)
                                d_iter+=1
        num_ano += 1        

        
# ## generate multi-layers subsurfaces, faults with and without anysotropy
def create_data_multi_layers(left, right, depth, grid, nx, nz, config_files=None, config_names=['dd','slm'], ratio=3, nb_elec=96, number_init=10, inf_res=10, sup_res=1000, area=1, cmap='jet', noiseAbs=5e-5, seed=1
                            , noise_levels=[0.01, 0.03, 0.05], rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names, rhoa_datFile_names=rhoa_datFile_names
                            , rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=False, with_inv=True , inv_rho_file_names=inv_rho_file_names
                            , paraDepth=depth, quality=34.6, paraMaxCellSize=100, limits=None, zWeight=None, anisotropy=True, inject_fluid=True, cmin=None, cmax=None, tol_lim=0.1
                            , fix_marker_area=0.1):
    """ Generate the database on multi-layers subsurfaces with and without anisotropy """
    # elecs arrays
    schemes_list = set_config(config_files=config_files, config_names=config_names, nb_elec=nb_elec)
    scheme = schemes_list[0]
        
    # subsurface
    world = mt.createWorld(start=[-left, 0], end=[right, -depth], marker=0, area=100)
    #homo_res = np.random.uniform(inf_res, sup_res, size=number_init)
    ## TODO make these parameters not manuel entring.
    add_scheme_nodes(world, schemes_list[0])
    mesh = mt.createMesh(world, area=area/5, smooth=True)
    N_cells_homo = mesh.cellCount()
    depths=[depth/5, depth*2/5, depth*2/5]
    depths_3l=[[depth/5, depth*2/5], [depth/3.5, depth/1.6], [depth/3.5, depth*3/4]]
    depths_4l=[[depth/5, depth/2, depth/1.4], [depth/5, depth*2/5, depth*3/4], [depth/6, depth/2.5, depth*3/4]]
    rand_vars = [0.5, 0.5, 1, 1, 1.5]
    rand_vars_3l = [0.5, 0.5, 0.8, 1, 1, 1.5]
    rand_vars4l = [0.3, 0.5, 0.8, 1, 1, 1.5, 1.5]
    offset_angles = [0, 6, 8]
    fault_shifts = [-8, -10, -12, -15, 15, 12, 10, 8]
    #cmin, cmax = inf_res, sup_res

    # Generate data
    for i in range(number_init):
        # seed a new random
        np.random.seed(random.randint(0, 2**32-1))
        print_status()
        # homogenous resistivity 
        res = random.randint(inf_res, sup_res)
        rhomap = [[0, res]]
        if anisotropy:
            # simulate and inverse by group
            meshes = []
            rhomaps = []
            rho_names = []
            # single layer with anysotropy due to different resistivity in porosity
            meshes.append(mesh)
            resistivity = add_anisotropy(mesh, N_cells_homo, rhomap, rho_poro=[random.randint(inf_res, sup_res)], porosity=[np.random.rand()])
            rhomaps.append(resistivity)
            rho_names.append('homo_res_'+str(round(res,2))+'_with_anysotropy_'+str(i)+'_rs_'+str(np.random.randint(1, 100000)))
            # potatoid body
            left_lim, right_lim, top_lim, bot_lim = int(-left+left*0.02), int(right-right*0.02), int(-depth*0.05), int(-depth+depth*0.1)
            center = [random.randint(left_lim, right_lim), random.randint(bot_lim, top_lim)]
            radius = [abs(abs(center[0])-min(-left_lim, right_lim))*0.9, min(np.abs(center[1] - np.array([top_lim, bot_lim])))*0.8]
            # Embed patatoid body in a one layer subsurfaces with anysotropy
            one_layers_pot, poly = add_patatoid(world, num_points=20, radius=radius, noise_var=0.5, center=center, marker=1, to_mesh=False, area=25, nodes=10)
            add_scheme_nodes(one_layers_pot, scheme)
            mesh_1layer_pot = mt.createMesh(one_layers_pot, area=area/5, smooth=True)
            meshes.append(mesh_1layer_pot)
            N_cells_1l_pot = mesh_1layer_pot.cellCount()
            rhomap_1l_pot = [[0, np.random.uniform(inf_res, sup_res)], [1, np.random.uniform(inf_res, sup_res)]]
            resistivity = add_anisotropy(mesh_1layer_pot, N_cells_1l_pot, rhomap_1l_pot, rho_poro=np.random.uniform(inf_res, sup_res, 2).tolist(), porosity=np.random.rand(2).tolist())
            rhomaps.append(resistivity)
            rho_names.append('one_layer_with_embed_potato'+'_anysotropy_'+str(i)+'_rs_'+str(np.random.randint(1, 100000)))
            # Embed patatoid body in a two layers subsurfaces with anysotropy
            two_layer_pot = make_two_layer(world, left, right, depth, control_points, random.choice(depths), nb_cp, rand_var=random.choice(rand_vars), tol_lim=tol_lim, marker=1, area=area, fault_shift=None, fault_range=None)
            two_layer_pot, poly = add_patatoid(two_layer_pot, num_points=20, radius=radius, noise_var=0.5, center=center, marker=2, to_mesh=False, area=25, nodes=10)
            add_scheme_nodes(two_layer_pot, scheme)
            mesh_2layer_pot = mt.createMesh(two_layer_pot, area=area/5, smooth=True)
            if np.random.rand() > 0.5:
                mesh_2layer_pot = fix_PLCmarker_to_mesh(mesh_2layer_pot, poly, marker=2, area=fix_marker_area)
            meshes.append(mesh_2layer_pot)
            N_cells_2l_pot = mesh_2layer_pot.cellCount()
            rhomap_2l_pot = [[0, np.random.uniform(inf_res, sup_res)], [1, np.random.uniform(inf_res, sup_res)], [2, np.random.uniform(inf_res, sup_res)]]
            rho_names.append('two_layers_with_embed_potato'+'_anysotropy_'+str(i)+'_rs_'+str(np.random.randint(1, 100000)))
            resistivity = add_anisotropy(mesh_2layer_pot, N_cells_2l_pot, rhomap_2l_pot, rho_poro=np.random.uniform(inf_res, sup_res, 3).tolist(), porosity=np.random.rand(3).tolist())
            rhomaps.append(resistivity)
            # Embed patatoid body in a three layers subsurfaces with anysotropy
            three_layer_pot = make_three_layers(world, left, right, depth, control_points, [depth/5, depth*2/5], nb_cp, rand_var=random.choice(rand_vars_3l), tol_lim=tol_lim, markers=[1, 2], area=area, same_shape=False)
            three_layer_pot, poly = add_patatoid(three_layer_pot, num_points=20, radius=radius, noise_var=0.5, center=center, marker=3, to_mesh=False, area=25, nodes=10)
            add_scheme_nodes(three_layer_pot, scheme)
            mesh_3layer_pot = mt.createMesh(three_layer_pot, area=area/5, smooth=True)
            mesh_3layer_pot = fix_PLCmarker_to_mesh(mesh_3layer_pot, poly, marker=3, area=fix_marker_area)
            meshes.append(mesh_3layer_pot)
            N_cells_3l_pot = mesh_3layer_pot.cellCount()
            rhomap_3l_pot = [[0, np.random.uniform(inf_res, sup_res)], [1, np.random.uniform(inf_res, sup_res)], [2, np.random.uniform(inf_res, sup_res)], [3, np.random.uniform(inf_res, sup_res)]]
            rho_names.append('three_layers_with_embed_potato'+'_anysotropy_'+str(i)+'_rs_'+str(np.random.randint(1, 100000)))
            resistivity = add_anisotropy(mesh_3layer_pot, N_cells_3l_pot, rhomap_3l_pot, rho_poro=np.random.uniform(inf_res, sup_res, 4).tolist(), porosity=np.random.rand(4).tolist())
            rhomaps.append(resistivity)
            # run the simulation and inversion for the group
            for k, anyso_mesh in enumerate(meshes):
                save_rho_data(name=rho_names[k], mesh=anyso_mesh, resistivity=rhomaps[k], grid=grid, nx=nx, nz=nz, resIsRhomap=False, rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names
                            , save_img=True, cmap=cmap, cmin=cmin, cmax=cmin)
                simulate_accros_configurations(config_names, noise_levels, anyso_mesh, grid, nx, nz, area, schemes_list, res_map=rhomaps[k], rho_name=rho_names[k], mesh_on=False, noiseAbs=1e-6, seed=seed
                                            , rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=True, with_inv=True 
                                            , inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits
                                            , zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)
            del meshes, rhomaps, rho_names, mesh_1layer_pot, mesh_3layer_pot, three_layer_pot, mesh_2layer_pot, one_layers_pot, two_layer_pot, N_cells_2l_pot, rhomap_2l_pot, resistivity, poly
            print_status()
        # simulate the multi-layers data
        for j in range(ratio):
            # seed a new random
            #np.random.seed(random.randint(0, 2**32-1))
            # simulate and inverse by group
            meshes = []
            rhomaps = []
            rho_names = []
            N_cells_mls = []
            # 2 layers resistivities
            rhomap_l2 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)]]
            rhomaps.append(rhomap_l2)
            world_l2 = make_two_layer(world, left, right, depth, control_points, random.choice(depths), nb_cp, rand_var=random.choice(rand_vars), tol_lim=tol_lim, marker=1, area=area, fault_shift=None, fault_range=None)
            mesh_l2 = mt.createMesh(world_l2, area=area/5, smooth=True)
            meshes.append(mesh_l2)
            rho_names.append('two_layer_top_' + str(round(rhomap_l2[0][1],2))+'_bottom_' + str(round(rhomap_l2[1][1],2))+'_'+str(i)+'_r'+str(j)+'_rs_'+str(np.random.randint(1, 100000)))
            N_cells_mls.append(mesh_l2.cellCount()) 
            # 3 layers resistivities with differents lithologies
            rhomap_l3 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)], [2, random.randint(inf_res, sup_res)]]
            rhomaps.append(rhomap_l3)
            world_l3 = make_three_layers(world, left, right, depth, control_points, random.choice(depths_3l), nb_cp, rand_var=random.choice(rand_vars_3l), tol_lim=tol_lim, markers=[1, 2], area=area, same_shape=False)
            rho_names.append('three_layer_top_' + str(round(rhomap_l3[0][1],2))+'_mid_'+str(round(rhomap_l3[1][1],2))+'_bottom_' + str(round(rhomap_l3[2][1],2))+'_'+str(i)+'_r'+str(j)+'_rs_'+str(np.random.randint(1, 100000)))
            mesh_l3 = mt.createMesh(world_l3, area=area/5, smooth=True)
            meshes.append(mesh_l3)
            N_cells_mls.append(mesh_l3.cellCount())
            # add anysotropy, simulate and inverse
            for k, rmesh in enumerate(meshes):
                save_rho_data(name=rho_names[k], mesh=rmesh, resistivity=rhomaps[k], grid=grid, nx=nx, nz=nz, resIsRhomap=True, rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names
                            , save_img=True, cmap=cmap)
                simulate_accros_configurations(config_names, noise_levels, rmesh, grid, nx, nz, area, schemes_list, res_map=rhomaps[k], rho_name=rho_names[k], mesh_on=False, noiseAbs=1e-6, seed=seed
                                            , rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=True, with_inv=True 
                                            , inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits
                                            , zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)
                if anisotropy:
                    rho_poro = np.random.randint(inf_res, sup_res, k+2).tolist()
                    resistivity = add_anisotropy(rmesh, N_cells_mls[k], rhomaps[k], rho_poro=rho_poro, porosity=list(np.random.rand(k+2)), interpolate=True)
                    rho_name = rho_names[k] + '_anysotropy'
                    save_rho_data(name=rho_name, mesh=rmesh, resistivity=resistivity, grid=grid, nx=nx, nz=nz, resIsRhomap=False, rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names
                                , save_img=True, cmin=cmin, cmax=cmin, cmap=cmap)
                    simulate_accros_configurations(config_names, noise_levels, rmesh, grid, nx, nz, area, schemes_list, res_map=resistivity, rho_name=rho_name, mesh_on=False, noiseAbs=1e-6, seed=seed
                                            , rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=True, with_inv=True 
                                            , inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits
                                            , zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)
            
            save_db_indexes_file(csv_file=csv_file, rho_file_names=rho_file_names, config_names=config_names, noise_levels=noise_levels, with_inv=True, save_img=True, verbose=True)
            del meshes, rhomaps, rho_names, N_cells_mls, mesh_l2, mesh_l3, world_l2, world_l3, rhomap_l2, rhomap_l3
            print_status()
            if j==0:
                # seed a new random
                np.random.seed(random.randint(0, 2**32-1))
                ## simulate resistivity constrast due to a fluid
                if inject_fluid:
                    # injection parameters
                    nb_inject = np.random.randint(1, 6)
                    sigmas_vars = np.arange(5, 15, 20)
                    mu_x = np.random.randint(-left, right, nb_inject)
                    mu_y = np.random.randint(-depth, 0, nb_inject)
                    sigma_x = np.random.choice(sigmas_vars, nb_inject)
                    sigma_y = np.random.choice(sigmas_vars, nb_inject)
                    cor_coef = np.random.rand(nb_inject) 
                    weights = np.random.rand(nb_inject)
                    weights = weights / np.sum(weights)
                    # simulate and inverse by group
                    meshes = []
                    rhomaps = []
                    rho_names = []
                    # 2 layers resistivities
                    rhomap_l2 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)]]
                    rhomaps.append(rhomap_l2)
                    world_l2 = make_two_layer(world, left, right, depth, control_points, random.choice(depths), nb_cp, rand_var=random.choice(rand_vars), tol_lim=tol_lim, marker=1, area=area, fault_shift=None, fault_range=None)
                    mesh_l2 = mt.createMesh(world_l2, area=area/5, smooth=True)
                    meshes.append(mesh_l2)
                    rho_names.append('two_layer_top_' + str(round(rhomap_l2[0][1],2))+'_bottom_' + str(round(rhomap_l2[1][1],2))+'_'+str(i)+'_r'+str(j)+'_rs_'+str(np.random.randint(1, 100000)))
                    # 3 layers resistivities with differents lithologies
                    rhomap_l3 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)], [2, random.randint(inf_res, sup_res)]]
                    rhomaps.append(rhomap_l3)
                    world_l3 = make_three_layers(world, left, right, depth, control_points, random.choice(depths_3l), nb_cp, rand_var=random.choice(rand_vars_3l), tol_lim=tol_lim, markers=[1, 2], area=area, same_shape=False)
                    rho_names.append('three_layer_top_' + str(round(rhomap_l3[0][1],2))+'_mid_'+str(round(rhomap_l3[1][1],2))+'_bottom_' + str(round(rhomap_l3[2][1],2))
                                     +'_'+str(i)+'_r'+str(j)+'_rs_'+str(np.random.randint(1, 100000)))
                    mesh_l3 = mt.createMesh(world_l3, area=area/5, smooth=True)
                    meshes.append(mesh_l3)
                    # 4 layers resistivities with fluid injection
                    rhomap_l4 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)], [2, random.randint(inf_res, sup_res)], [3, random.randint(inf_res, sup_res)]]
                    rhomaps.append(rhomap_l4)
                    world_l4 = make_four_layers(world, left, right, depth, control_points, random.choice(depths_4l), nb_cp, rand_var=random.choice(rand_vars4l), tol_lim=tol_lim
                                                , markers=[1, 2, 3], area=area, same_shape=True)
                    mesh_l4_inject = mt.createMesh(world_l4, area=area/5, smooth=True)
                    meshes.append(mesh_l4_inject)
                    rho_names.append('four_layer_top_' + str(round(rhomap_l4[0][1],2))+'_mid_'+str(round(rhomap_l4[1][1],2))+'_bottom_' + str(round(rhomap_l4[2][1],2))+'_'
                                     +str(i)+'_r'+str(j)+'_rs_'+str(np.random.randint(1, 100000)))
                    for k, mesh_inj in enumerate(meshes):
                        # cells centers in the mesh
                        cell_centers = mesh_inj.cellCenters()
                        # gausian anomaly 
                        res = mix_gaussian(cell_centers, mu_x, mu_y, sigma_x, sigma_y, cor_coef, weights)
                        # inject the gausian into mesh
                        if k==0:
                            inject_into_mesh(mesh_inj, res, rhomaps[k], inj_res=random.randint(inf_res, sup_res), enclosed=[0])
                        else:
                            inject_into_mesh(mesh_inj, res, rhomaps[k], inj_res=random.randint(inf_res, sup_res), enclosed=random.choice([[1,2], [0,1,2]]))
                        rho_name = rho_names[k] + '_fluidInject'
                        save_rho_data(name=rho_name, mesh=mesh_inj, resistivity=res, grid=grid, nx=nx, nz=nz, resIsRhomap=False, rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names
                                    , save_img=True, cmin=cmin, cmax=cmin, cmap=cmap)
                        simulate_accros_configurations(config_names, noise_levels, mesh_inj, grid, nx, nz, area, schemes_list, res_map=res, rho_name=rho_name, mesh_on=False, noiseAbs=1e-6, seed=seed
                                                , rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=True, with_inv=True 
                                                , inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits
                                                , zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)
                    del meshes, rhomaps, rho_names, mesh_l2, mesh_l3, world_l2, world_l3, rhomap_l2, rhomap_l3, mesh_l4_inject, world_l4, rhomap_l4, res
                ## subsurfaces with some tectonic events
                # simulate and inverse by group
                meshes = []
                rhomaps = []
                rho_names = []
                N_cells_mls = []
                # creating fault ranges
                init_fault = random.randint(5, nb_cp-10)
                fault_ranges = [list(np.arange(init_fault,nb_cp)),  list(np.arange(init_fault, random.randint(init_fault+5, nb_cp)))]          
                # 3 layers resistivities with shame lithology having some tectonic events
                rhomap_l3 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)], [2, random.randint(inf_res, sup_res)]]
                rhomaps.append(rhomap_l3)
                world_l3 = make_three_layers(world, left, right, depth, control_points, random.choice(depths_3l), nb_cp, rand_var=random.choice(rand_vars_3l), tol_lim=tol_lim, markers=[1, 2]
                                            , offset_angle=random.choice(offset_angles), fault_shift=random.choice(fault_shifts), fault_range=random.choice(fault_ranges), area=area, same_shape=True)
                rho_names.append('three_layer_tecto_top_' + str(round(rhomap_l3[0][1],2))+'_mid_'+str(round(rhomap_l3[1][1],2))+'_bottom_' + str(round(rhomap_l3[2][1],2))
                                 +'_'+str(i)+'_r'+str(j)+'_rs_'+str(np.random.randint(1, 100000)))
                mesh_l3_fault = mt.createMesh(world_l3, area=area, smooth=True)
                meshes.append(mesh_l3_fault)
                N_cells_mls.append(mesh_l3_fault.cellCount()) 
                
                # 4 layers resistivities with shame lithology having some tectonic events
                rhomap_l4 = [[0, random.randint(inf_res, sup_res)], [1, random.randint(inf_res, sup_res)], [2, random.randint(inf_res, sup_res)], [3, random.randint(inf_res, sup_res)]]
                rhomaps.append(rhomap_l4)
                world_l4 = make_four_layers(world, left, right, depth, control_points, random.choice(depths_4l), nb_cp, rand_var=random.choice(rand_vars4l), tol_lim=tol_lim, markers=[1, 2, 3]
                                            , offset_angle=random.choice(offset_angles), fault_shift=random.choice(fault_shifts), fault_range=random.choice(fault_ranges), area=area, same_shape=True)
                rho_names.append('four_layer_tecto_top_' + str(round(rhomap_l4[0][1],2))+'_mid_'+str(round(rhomap_l4[1][1],2))+'_bottom_' + str(round(rhomap_l4[2][1],2))+'_'
                                 +str(i)+'_r'+str(j)+'_rs_'+str(np.random.randint(1, 100000)))
                mesh_l4_fault = mt.createMesh(world_l4, area=area/5, smooth=True)
                meshes.append(mesh_l4_fault)
                N_cells_mls.append(mesh_l4_fault.cellCount()) 
                for k, mesh_tecto in enumerate(meshes):
                    save_rho_data(name=rho_names[k], mesh=mesh_tecto, resistivity=rhomaps[k], grid=grid, nx=nx, nz=nz, resIsRhomap=True, rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names
                                , save_img=True, cmap=cmap)
                    simulate_accros_configurations(config_names, noise_levels, mesh_tecto, grid, nx, nz, area, schemes_list, res_map=rhomaps[k], rho_name=rho_names[k], mesh_on=False, noiseAbs=1e-6, seed=seed
                                                , rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=True, with_inv=True 
                                                , inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits
                                                , zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)
                    if anisotropy:
                        rho_poro = np.random.randint(inf_res, sup_res, k+3).tolist()
                        resistivity = add_anisotropy(mesh_tecto, N_cells_mls[k], rhomaps[k], rho_poro=rho_poro, porosity=list(np.random.rand(k+3)), interpolate=True)
                        rho_name = rho_names[k] + '_anysotropy'
                        save_rho_data(name=rho_name, mesh=mesh_tecto, resistivity=resistivity, grid=grid, nx=nx, nz=nz, resIsRhomap=False, rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names
                                    , save_img=True, cmin=cmin, cmax=cmin, cmap=cmap)
                        simulate_accros_configurations(config_names, noise_levels, mesh_tecto, grid, nx, nz, area, schemes_list, res_map=resistivity, rho_name=rho_name, mesh_on=False, noiseAbs=1e-6, seed=seed
                                                , rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=True, with_inv=True 
                                                , inv_rho_file_names=inv_rho_file_names, paraDepth=paraDepth, quality=quality, paraMaxCellSize=paraMaxCellSize, limits=limits
                                                , zWeight=zWeight, cmin=cmin, cmax=cmax, cmap=cmap)
                del meshes, rhomaps, rho_names, N_cells_mls, mesh_l3_fault, mesh_l4_fault, world_l4, rhomap_l4, resistivity
                
        save_db_indexes_file(csv_file=csv_file, rho_file_names=rho_file_names, config_names=config_names, noise_levels=noise_levels, with_inv=True, save_img=True, verbose=True)        