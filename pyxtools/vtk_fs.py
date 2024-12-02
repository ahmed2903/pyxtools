import math 
import numpy as np 
from scipy.interpolate import RegularGridInterpolator

import vtk
from vtk.util import numpy_support

def RotateVecs(data, axis, angle):
	shp = data.shape[0:3]
	angle = np.deg2rad(angle)
	if axis == 0:
		rot = X_Rot(angle)
	elif axis == 1:
		rot = Y_Rot(angle)
	elif axis == 2:
		rot = Z_Rot(angle)
	dataflat = np.empty((shp[0]*shp[1]*shp[2],3), dtype=np.double)
	dataflat[:,0] = data[:,:,:,0].flatten()
	dataflat[:,1] = data[:,:,:,1].flatten()
	dataflat[:,2] = data[:,:,:,2].flatten()
	data_rotated_flat = np.dot(dataflat, rot)
	data_rotated = data_rotated_flat.reshape(*shp,3)
	return data_rotated

def Transform(array, distance, binx, biny, binz, twotheta, dtheta, kind):
    
    """
    Bonsu Code of Cooridnate Transformation
    """
    shp = np.array(array.shape, dtype=np.int64)
    
    # Distance to Detector
    R = distance 
    
    # Detector Pixel Size
    dx = 50 * 10.0**(-6)
    dy = 50 * 10.0**(-6)
    
    # Wavelength
    waveln = 0.1377 * 10.0**(-9)
    
    k = 2.0*math.pi/waveln
    

    phi = 0.0
    dphi = 0.0
    
    dpx = binx*dx/R
    dpy = biny*dy/R
    
    Q = k * np.array([math.cos(phi)*math.sin(twotheta),math.sin(phi), (math.cos(phi)*math.cos(twotheta) - 1.0) ], dtype=np.double)
    
    Qmag = math.sqrt(np.dot(Q,Q))
    
    dQdx = [-math.sin(phi)*math.sin(twotheta), math.cos(phi), -math.sin(phi)*math.cos(twotheta)]
    
    dQdy = [math.cos(phi)*math.cos(twotheta), 0.0, -math.cos(phi)*math.sin(twotheta)]
    
    dQdtheta = [math.cos(phi)*math.cos(twotheta) - 1.0, 0.0, -math.cos(phi)*math.sin(twotheta)]
    
    dQdphi= [-math.sin(phi)*math.sin(twotheta), math.sin(phi) - 1.0, -math.sin(phi)*math.cos(twotheta)]
    
    astar = k*dpx * np.array(dQdx, dtype=np.double)
    
    bstar = k*dpy * np.array(dQdy, dtype=np.double)
    
    cstar = k*dtheta * binz * np.array(dQdtheta, dtype=np.double)
  
    v = np.dot(astar,np.cross(bstar,cstar))
    
    if kind == 'Real-space':
        a = 2.0*math.pi*np.cross(bstar,cstar)/v
        b = 2.0*math.pi*np.cross(cstar,astar)/v
        c = 2.0*math.pi*np.cross(astar,bstar)/v
        mx = 1.0/float(shp[0])
        my = 1.0/float(shp[1])
        mz = 1.0/float(shp[2])
        nmeter = 10.0**(9)
    else:
        a = astar
        b = bstar
        c = cstar
        mx = 1.0
        my = 1.0
        mz = 1.0
        nmeter = 10.0**(-9)
        
    T = np.vstack((a,b,c)).T
    vtk_dataset = vtk.vtkStructuredGrid()
    flat_data_amp = (np.abs(array)).transpose(2,1,0).flatten()
    flat_data_phase = (np.angle(array)).transpose(2,1,0).flatten()
    vtk_scalararray_amp = numpy_support.np_to_vtk(flat_data_amp)
    vtk_scalararray_phase = numpy_support.np_to_vtk(flat_data_phase)
    vtk_points = vtk.vtkPoints()
    
    coordarray = np.zeros((array.size,3), dtype=np.double)
    vtk_points.SetDataTypeToDouble()
    vtk_points.SetNumberOfPoints(array.size)
    
    X = np.zeros((shp[2]*shp[1]*shp[0],3), dtype=np.double)
    X[:,0] = mx*np.tile(np.arange(shp[0])[::-1], shp[1]*shp[2])
    X[:,1] = my*np.tile(np.repeat(np.arange(shp[1]), shp[0]), shp[2])
    X[:,2] = mz*np.repeat(np.arange(shp[2]), shp[0]*shp[1])
    coordarray[:] = nmeter * np.dot(T,X.T).T
    
    return coordarray


def InterpolatedScalarDataset(InputDataSet, grid, irange, cbounds):
    def TPObserver(obj, event):
        pass
    bounds=list(InputDataSet.GetBounds())
    RegGrid=vtk.vtkShepardMethod()
    RegGrid.SetMaximumDistance(irange)
    RegGrid.SetSampleDimensions(grid)
    RegGrid.SetModelBounds(cbounds)
    tp = vtk.vtkTrivialProducer()
    tp.SetOutput(InputDataSet)
    tp.SetWholeExtent(InputDataSet.GetExtent())
    tp.AddObserver(vtk.vtkCommand.ErrorEvent, TPObserver)
    ## RegGrid.SetInputData(InputDataSet)
    RegGrid.SetInputConnection(tp.GetOutputPort())
    RegGrid.GetInputInformation().Set(vtk.vtkStreamingDemandDrivenPipeline.UNRESTRICTED_UPDATE_EXTENT(),1)
    RegGrid.Update()
    return RegGrid.GetOutput()

def InterpolateObject(data, coords, gridsize, cbounds, irange):
    """
    Bonsu Code of Interpolation
    """
    
    shp = np.array(data.shape, dtype=np.int64)
    vtk_coordarray = numpy_support.np_to_vtk(coords)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToDouble()
    vtk_points.SetNumberOfPoints(data.size)
    vtk_points.SetData(vtk_coordarray)
    ## amp
    flat_data_amp = (np.abs(data)).transpose(2,1,0).flatten()
    vtk_data_array_amp = numpy_support.np_to_vtk(flat_data_amp)
    image_amp = vtk.vtkStructuredGrid()
    image_amp.SetPoints(vtk_points)
    image_amp.GetPointData().SetScalars(vtk_data_array_amp)
    image_amp.SetDimensions(shp)
    image_amp.Modified()
    """
    ## phase
    flat_data_phase = (np.angle(data)).transpose(2,1,0).flatten()
    vtk_data_array_phase = np_support.np_to_vtk(flat_data_phase)
    image_phase = vtk.vtkStructuredGrid()
    image_phase.SetPoints(vtk_points)
    image_phase.GetPointData().SetScalars(vtk_data_array_phase)
    image_phase.SetDimensions(shp)
    image_phase.Modified()
    """
    ## bounds and scale
    use_cbounds = False
    for i in range(6):
        if cbounds[i] != 0.0:
            use_cbounds = True
    if use_cbounds:
        bds = cbounds
    else:
        bds = list(image_amp.GetBounds())
    print(bds)
    interp_image_amp = InterpolatedScalarDataset(image_amp, gridsize, irange, bds)
    #interp_image_phase = InterpolatedScalarDataset(image_phase, gridsize, irange, bds)
    dims = interp_image_amp.GetDimensions()
    array_amp_flat = numpy_support.vtk_to_np(interp_image_amp.GetPointData().GetScalars())
    array_amp = np.reshape(array_amp_flat, dims[::-1]).transpose(2,1,0)
    #array_phase_flat = np_support.vtk_to_np(interp_image_phase.GetPointData().GetScalars())
    #array_phase = np.reshape(array_phase_flat, dims[::-1]).transpose(2,1,0)
    ##
    #array = array_amp * (np.cos(array_phase) + 1j * np.sin(array_phase))
    
    return array_amp

def AffineTransform(coords, translate, scale, rotate):
	vtk_coordarray = numpy_support.np_to_vtk(coords)
	vtk_points = vtk.vtkPoints()
	vtk_points.SetDataTypeToDouble()
	vtk_points.SetNumberOfPoints(coords.shape[0])
	vtk_points.SetData(vtk_coordarray)
	image_amp = vtk.vtkStructuredGrid()
	image_amp.SetPoints(vtk_points)
	image_amp.Modified()
	Transform = vtk.vtkTransform()
	Transform.Translate([translate[0],translate[1],translate[2]])
	Transform.Scale([scale[0],scale[1],scale[2]])
	Transform.RotateX(rotate[0])
	Transform.RotateY(rotate[1])
	Transform.RotateZ(rotate[2])
	Transform.Modified()
	TransFilter=vtk.vtkTransformFilter()
	TransFilter.SetTransform(Transform)
	TransFilter.SetInputData(image_amp)
	TransFilter.UpdateWholeExtent()
	TransFilter.Modified()
	new_vtk_points = TransFilter.GetOutput(0).GetPoints()
	new_coords = numpy_support.vtk_to_np(new_vtk_points.GetData())
	return new_coords
def MakePlaneCoords(shp, spacing):
	coords = np.zeros((shp[2]*shp[1]*shp[0],3), dtype=np.double)
	coords[:,0] = spacing[0]*np.tile(np.arange(shp[0]), shp[1]*shp[2])
	coords[:,1] = spacing[1]*np.tile(np.repeat(np.arange(shp[1]), shp[0]), shp[2])
	coords[:,2] = spacing[2]*np.repeat(np.arange(shp[2]), shp[0]*shp[1])
	return coords

def RotateVectorField(data, gridsize=[100,100,100], rot_axis=0, rot_angle=0, cbounds = [0,0,0,0,0,0], irange = 0.001, spacing = [1,1,1]):
	shp = data.shape[0:3]
	data_vec_rot = RotateVecs(data, rot_axis, rot_angle)
	coords = MakePlaneCoords(shp, spacing)
	rot = [0,0,0]
	rot[rot_axis] = rot_angle
	newcoords = AffineTransform(coords, [0,0,0], [1,1,1], rot)
	data_vec_rot_x = np.squeeze(data_vec_rot[:,:,:,0])
	data_vec_rot_y = np.squeeze(data_vec_rot[:,:,:,1])
	data_vec_rot_z = np.squeeze(data_vec_rot[:,:,:,2])
	if gridsize is None:
		gridsize = [int(newcoords[:,0].max() - newcoords[:,0].min()),\
					int(newcoords[:,1].max() - newcoords[:,1].min()),\
					int(newcoords[:,2].max() - newcoords[:,2].min())]
	newdata_vec_rot_x = InterpolateObject(data_vec_rot_x, newcoords, gridsize, cbounds, irange)
	newdata_vec_rot_y = InterpolateObject(data_vec_rot_y, newcoords, gridsize, cbounds, irange)
	newdata_vec_rot_z = InterpolateObject(data_vec_rot_z, newcoords, gridsize, cbounds, irange)
	newdata = np.stack((newdata_vec_rot_x, newdata_vec_rot_y, newdata_vec_rot_z), axis = -1)
	return newdata
	
def RotateMask(mask, gridsize=[100,100,100], rot_axis=0, rot_angle=0, cbounds = [0,0,0,0,0,0], irange = 0.001, spacing = [1,1,1]):
	shp = mask.shape
	coords = MakePlaneCoords(shp, spacing)
	rot = [0,0,0]
	rot[rot_axis] = rot_angle
	newcoords = AffineTransform(coords, [0,0,0], [1,1,1], rot)
	if gridsize is None:
		gridsize = [int(newcoords[:,0].max() - newcoords[:,0].min()),\
					int(newcoords[:,1].max() - newcoords[:,1].min()),\
					int(newcoords[:,2].max() - newcoords[:,2].min())]
	newmask = InterpolateObject(mask, newcoords, gridsize, cbounds, irange)