from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
import pyqtgraph as pg
import numpy as np
from matplotlib.pylab import imread #scipy.misc.imread is depreciated!
import os
import pyqtgraph.dockarea
from qtpy import QtCore, QtWidgets
from pyqtgraph.graphicsItems.ROI import EllipseROI
import h5py

class Gauss2DFitImgView(DataBrowserView):

    name = 'gauss2d_fit_img'
    
    def setup(self):
        
        self.ui = self.dockarea = pyqtgraph.dockarea.DockArea()
        self.imview = pg.ImageView()
        self.imview.getView().invertY(True) # uper left origin
        self.dockarea.addDock(name='Full Image', widget=self.imview)

        self.imview_roi = pg.ImageView()
        self.imview_roi.getView().invertY(True) # upper left origin
        self.dockarea.addDock(name='ROI Image', widget=self.imview_roi)

        self.imview_gauss = pg.ImageView()
        self.imview_gauss.getView().invertY(True) # upper left origin
        self.dockarea.addDock(name='Gauss Fit Image', widget=self.imview_gauss)
        
        
        self.rect_roi = pg.RectROI([20, 20], [100, 100], pen=(0,9))
        self.imview.getView().addItem(self.rect_roi)        
        self.rect_roi.sigRegionChanged[object].connect(self.on_change_rect_roi)

        self.info_label = QtWidgets.QLabel()
        self.dockarea.addDock(name='info', widget=self.info_label)

        
    def on_change_data_filename(self, fname):
        
        try:
            self.data = imread(fname)
            if len(self.data.shape) == 3:
                self.data = self.data.sum(axis=2)
                
        except Exception as err:
            self.data = np.zeros((10,10))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        self.imview.setImage(self.data, axes=dict(x=1,y=0))
        self.on_change_rect_roi()

        
    def is_file_supported(self, fname):
        _, ext = os.path.splitext(fname)
        return ext.lower() in ['.png', '.tif', '.tiff', '.jpg']


    def on_change_rect_roi(self, roi=None):
        # pyqtgraph axes are x,y, but data is stored in (frame, y,x, time)
        #roi_slice, roi_tr = self.rect_roi.getArraySlice(self.data, self.imview.getImageItem(), axes=(1,0))
        #self.rect_plotdata.setData(self.spec_x_array, self.hyperspec_data[roi_slice].mean(axis=(0,1))+1)
        
        self.roi_img = self.rect_roi.getArrayRegion(data=self.data, img=self.imview.getImageItem(), axes=(1,0)) 
        
        #print(self.roi_img.shape)
        
        self.imview_roi .setImage(self.roi_img, axes=dict(x=1,y=0)) #, axes=dict(x=1,y=0))
        #print("roi_slice", roi_slice)
        
        p,success = fitgaussian(self.roi_img)
        
        height, x, y, width_x, width_y, angle = p
        print(p)
        
        if success:
            
            fwhm_x = 2*np.sqrt(2*np.log(2)) * width_x
            fwhm_y = 2*np.sqrt(2*np.log(2)) * width_y
            
            self.info_label.setText(
                "height:{}, x: {}, y: {}, width_x: {}, width_y: {}, angle: {}".format(*p) +
                "\nfwhm {} {}".format(fwhm_x, fwhm_y)
                )
            
            ny,nx = self.roi_img.shape
            X,Y = np.mgrid[:ny,:nx]
            self.gaus_img = gaussian(*p)(X,Y)
            
            self.imview_gauss.setImage(self.gaus_img, axes=dict(x=1,y=0)) #, axes=dict(x=1,y=0))
            
            
            if not hasattr(self, 'iso_curve'):
                self.iso_curve = pg.IsocurveItem(data=self.gaus_img, level=0.5)#, pen, axisOrder)
                self.imview_roi.getView().addItem(self.iso_curve)
            else:
                print(height*0.5)
                self.iso_curve.setData(data=self.gaus_img.T, level=height*0.5)
                self.iso_curve.setParentItem(self.imview_roi.getImageItem())



#From https://gist.github.com/andrewgiessel/6122739
import scipy.optimize

def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x,y):
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        g = height*np.exp(
            -(((center_x-xp)/width_x)**2+
              ((center_y-yp)/width_y)**2)/2.)
        return g
    return rotgauss

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0.0


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y, angle)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = scipy.optimize.leastsq(errorfunction, params)
    return p,success
    
    
    
class Gauss2DFitAPD_MCL_2dSlowScanView(Gauss2DFitImgView):
    
    name = 'gauss2d_fit_img_apd_mcl'

    def on_change_data_filename(self, fname):
        try:
            self.dat = h5py.File(fname, 'r')
            self.data = np.array(self.dat['measurement/APD_MCL_2DSlowScan/count_rate_map'][0,:,:].T) # grab first frame
        except Exception as err:
            self.data = np.zeros((10,10))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        self.imview.setImage(self.data, axes=dict(x=1,y=0))
        self.on_change_rect_roi()
        
class Gauss2DFit_FiberAPD_View(Gauss2DFitImgView):
    
    name = 'gauss2d_fit_img_fiber_apd'

    def on_change_data_filename(self, fname):
        try:
            self.dat = h5py.File(fname, 'r')
            self.data = np.array(self.dat['measurement/fiber_apd_scan/count_rate_map'][0,:,:].T) # grab first frame
        except Exception as err:
            self.data = np.zeros((10,10))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        self.imview.setImage(self.data, axes=dict(x=1,y=0))
        self.on_change_rect_roi()