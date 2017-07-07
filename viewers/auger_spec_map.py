from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg
import pyqtgraph.dockarea as dockarea
from qtpy import QtWidgets, QtGui
from scipy import optimize

import sys
#sys.path.insert(0, '/home/dbdurham/foundry_scope/FoundryDataBrowser/viewers')
from .drift_correction import register_translation_hybrid, shift_subpixel

class AugerSpecMapView(DataBrowserView):
    
    name = 'auger_spec_map'
    
    def setup(self):
        
        self.settings.New('equalize_detectors', dtype=bool)
        self.settings.get_lq('equalize_detectors').add_listener(self.on_change_equalize_detectors)
        
        self.settings.New('drift_correct', dtype=bool)
        self.settings.get_lq('drift_correct').add_listener(self.on_change_drift_correct)
        
        self.settings.New('spectrum_over_ROI', dtype=bool)
        self.settings.get_lq('spectrum_over_ROI').add_listener(self.on_change_spectrum_over_ROI)
        
        # Subtract the B section (ke1_start through ke1_stop) by a power law fit
        self.settings.New('subtract_spectrum_background', dtype=bool)
        self.settings.get_lq('subtract_spectrum_background').add_listener(self.on_change_subtract_spectrum_background)
        
        self.settings.New('ke0_start', dtype=float)
        self.settings.New('ke0_stop', dtype=float)
        self.settings.New('ke1_start', dtype=float)
        self.settings.New('ke1_stop', dtype=float)
        
        #Math mode now updates automatically on change
        self.settings.New('math_mode', dtype=str, initial='A')
        self.settings.get_lq('math_mode').add_listener(self.on_change_math_mode)
        
        for lqname in ['ke0_start', 'ke0_stop', 'ke1_start', 'ke1_stop']:
            self.settings.get_lq(lqname).add_listener(self.on_change_ke_settings)
        
        self.ui = self.dockarea = dockarea.DockArea()
        
        self.dockarea.addDock(name='Settings', widget=self.settings.New_UI())
        
        # Spectrum plot
        self.graph_layout = pg.GraphicsLayoutWidget()        
        self.spec_plot = self.graph_layout.addPlot()
        #self.rect_plotdata = self.spec_plot.plot()
        #self.point_plotdata = self.spec_plot.plot(pen=(0,9))
        self.total_plotline = self.spec_plot.plot()
        self.dockarea.addDock(name='Spec Plot', widget=self.graph_layout)

        self.lr0 = pg.LinearRegionItem(values=[0,1], brush=QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
        self.lr1 = pg.LinearRegionItem(values=[2,3], brush=QtGui.QBrush(QtGui.QColor(255, 0, 0, 50)))
        
        for lr in (self.lr0, self.lr1):
            lr.setZValue(10)
            self.spec_plot.addItem(lr, ignoreBounds=True)
            lr.sigRegionChangeFinished.connect(self.on_change_regions)
            
        self.chan_plotlines = []
        for ii in range(7):
            color = pg.intColor(ii)
            self.chan_plotlines.append(
                self.spec_plot.plot([0], pen=color))
            
            
        # Images
        self.imview_sem0 = pg.ImageView()
        self.imview_sem0.getView().invertY(False) # lower left origin
        self.dockarea.addDock(name='SEM A Image', widget=self.imview_sem0)

        self.imview_sem1 = pg.ImageView()
        self.imview_sem1.getView().invertY(False) # lower left origin
        self.dockarea.addDock(name='SEM B Image', widget=self.imview_sem1)

        self.imview_auger = pg.ImageView()
        self.imview_auger.getView().invertY(False) # lower left origin
        self.dockarea.addDock(name='Auger Map', widget=self.imview_auger)
        self.im_auger = self.imview_auger.getImageItem()
        
        # Polygon ROI
        self.poly_roi = pg.PolyLineROI([[20,0], [20,20], [0,20], [20,0]], pen=(0,9), closed=True)
        #self.poly_roi = pg.RectROI([20, 20], [20, 20], pen=(0,9))
        #self.poly_roi = pg.CircleROI((0,0), (10,10) , movable=True, pen=(0,9))
        #self.poly_roi.addTranslateHandle((0.5,0.5))        
        self.imview_auger.getView().addItem(self.poly_roi)        
        self.poly_roi.sigRegionChanged[object].connect(self.on_change_roi)

    def is_file_supported(self, fname):
        return "auger_sync_raster_scan.h5" in fname

    def on_change_data_filename(self, fname=None):
        try:
            self.dat = h5py.File(fname, 'r')
            self.H = self.dat['measurement/auger_sync_raster_scan/']
            h = self.h_settings = self.H['settings'].attrs
            self.adc_map = np.array(self.H['adc_map'])
            self.ctr_map = np.array(self.H['ctr_map'])
            self.auger_map = np.array(self.H['auger_chan_map'], dtype=float)
            time_per_px = self.auger_map[:,:,:,:,8:9]* 25e-9 # units of 25ns converted to seconds
            self.auger_map = self.auger_map[:,:,:,:,0:7]/time_per_px # auger map now in Hz
            self.auger_sum_map = self.auger_map[:,:,:,:,0:7].mean(axis=4)
            
            self.ke = np.array(self.H['ke'])
            
            #scan_shape = self.adc_map.shape[:-1]
            
            # Calculate relative detector efficiencies
            self.calculate_detector_efficiencies()
            if self.settings['equalize_detectors']:
                self.auger_map /= self.det_eff
            
            self.update_spectrum_display()
            
            self.on_change_ke_settings()
            
            #self.update_display()
        except Exception as err:
            print(err)
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("Failed to load %s: %s" %(fname, err))
            raise(err)
    
    def on_change_drift_correct(self):
        
        if self.settings['drift_correct']:
            #### Phase 1: Pairwise Correction ####
            
            scan_shape = self.adc_map.shape[:-1]
            adc_chan = 0 # Default correction channel
            num_frames = scan_shape[0]  # Consider allowing correcting only over a range in case data has blank or incorrect frames
            
            # Prepare window function (Hann)
            win = np.outer(np.hanning(scan_shape[2]),np.hanning(scan_shape[3]))
            
            # Pairwise shifts
            shift = np.zeros((2, num_frames))
            for iFrame in range(1, num_frames):
                image = self.adc_map[iFrame-1,0,:,:,adc_chan]
                offset_image = self.adc_map[iFrame,0,:,:,adc_chan]
                shift[:,iFrame], error, diffphase = register_translation_hybrid(image*win, offset_image*win, exponent = 0.3, upsample_factor = 100)
            
            # Shifts are defined as [y, x] where y is shift of imaging location with respect to positive y axis, similarly for x
                
            shift_sum = np.sum(shift, axis=1)
            # Cumulative sum defines shift with respect to original image for each image
            # Maxima and minima in cumulative x and y shifts defines box within which all images have defined pixels
            shift_cumul = np.cumsum(shift, axis=1)
            
            # Determining coordinates of fully defined box for original image

            shift_cumul_y = shift_cumul[0,:]
            shift_cumul_x = shift_cumul[1,:]
            
            # NOTE: scan_shape indices 2, 3 correspond to y, x
            y1 = int(round(np.max(shift_cumul_y[shift_cumul_y >= 0])+0.001, 0))
            y2 = int(round(scan_shape[2] + np.min(shift_cumul_y[shift_cumul_y <= 0])-0.001, 0))
            x1 = int(round(np.max(shift_cumul_x[shift_cumul_x >= 0])+0.001, 0))
            x2 = int(round(scan_shape[3] + np.min(shift_cumul_x[shift_cumul_x <= 0])-0.001, 0))
            
            boxfd = np.array([y1,y2,x1,x2])
            boxdims = (boxfd[1]-boxfd[0], boxfd[3]-boxfd[2])
            
            # Shift images to align
            imstack = np.zeros((num_frames, scan_shape[1], scan_shape[2], scan_shape[3], 2))
            specstack = np.zeros((num_frames, scan_shape[1], scan_shape[2], scan_shape[3], self.auger_map.shape[4]))
            for iFrame in range(0, num_frames):
                # Shift adc map
                for iDet in range(0, imstack.shape[4]):
                    imstack[iFrame,0,:,:,iDet] = shift_subpixel(self.adc_map[iFrame,0,:,:,iDet], dx=shift_cumul_x[iFrame], dy=shift_cumul_y[iFrame])
                # Shift spectral data
                for iDet in range(0, self.auger_map.shape[4]):
                    specstack[iFrame,0,:,:,iDet] = shift_subpixel(self.auger_map[iFrame,0,:,:,iDet], dx=shift_cumul_x[iFrame], dy=shift_cumul_y[iFrame])
            
            # Keep only preserved data
            imstack = np.real(imstack[:,:,boxfd[0]:boxfd[1], boxfd[2]:boxfd[3],:])
            specstack = np.real(specstack[:,:,boxfd[0]:boxfd[1], boxfd[2]:boxfd[3],:])
            
            #### Phase 2: Running Average ####
            
            # Update the image shape
            imshape = imstack.shape
            imstack_run = imstack.copy()
            specstack_run = specstack.copy()
            
            # Prepare window function (Hann)
            win = np.outer(np.hanning(imshape[2]),np.hanning(imshape[3]))
            
            # Shifts to running average
            shift = np.zeros((2, num_frames))
            image = imstack[0,0,:,:,adc_chan]
            for iFrame in range(1, num_frames):
                offset_image = imstack[iFrame,0,:,:,adc_chan]
                # Calculate shift
                shift[:,iFrame], error, diffphase = register_translation_hybrid(image*win, offset_image*win, exponent = 0.3, upsample_factor = 100)
                # Perform shifts
                # Shift adc map
                for iDet in range(0, imstack.shape[4]):
                    imstack_run[iFrame,0,:,:,iDet] = shift_subpixel(imstack[iFrame,0,:,:,iDet], dx = shift[1,iFrame], dy = shift[0, iFrame])
                 # Shift spectral data
                for iDet in range(0, specstack.shape[4]):
                    specstack_run[iFrame,0,:,:,iDet] = shift_subpixel(specstack[iFrame,0,:,:,iDet], dx = shift[1,iFrame], dy = shift[0, iFrame])
                # Update running average
                image = (iFrame/(iFrame+1)) * image + (1/(iFrame+1)) * imstack_run[iFrame,0,:,:,adc_chan]
            # Shifts are defined as [y, x] where y is shift of imaging location with respect to positive y axis, similarly for x
            
            # Determining coordinates of fully defined box for original image

            shift_y = shift[0,:]
            shift_x = shift[1,:]
            
            # NOTE: scan_shape indices 2, 3 correspond to y, x
            y1 = int(round(np.max(shift_y[shift_y >= 0])+0.001, 0))
            y2 = int(round(imshape[2] + np.min(shift_y[shift_y <= 0])-0.001, 0))
            x1 = int(round(np.max(shift_x[shift_x >= 0])+0.001, 0))
            x2 = int(round(imshape[3] + np.min(shift_x[shift_x <= 0])-0.001, 0))
            
            boxfd = np.array([y1, y2, x1, x2])
            boxdims = (boxfd[1]-boxfd[0], boxfd[3]-boxfd[2])
            
            # Keep only preserved data
            self.adc_map = np.real(imstack_run[:,:,boxfd[0]:boxfd[1], boxfd[2]:boxfd[3],:])
            self.auger_map = np.real(specstack_run[:,:,boxfd[0]:boxfd[1], boxfd[2]:boxfd[3],:])
            
            # Display
            self.update_spectrum_display()
            
            self.on_change_ke_settings()
        
        
    def on_change_equalize_detectors(self):
        if self.settings['equalize_detectors']:
            self.auger_map /= self.det_eff
        else:
            self.auger_map *= self.det_eff
        
        # Update spectrum displays
        self.update_spectrum_display()
        
        # Update maps
        self.on_change_ke_settings()
        
        pass
    
    def on_change_ke_settings(self):
        
        
        print ("on_change_ke_settings")
        S = self.settings
        print('ke shape', self.ke.shape)
        ke_map0 = (S['ke0_start'] < self.ke) * (self.ke < S['ke0_stop'])
        ke_map1 = (S['ke1_start'] < self.ke) * (self.ke < S['ke1_stop'])
        
        # KE of shape n_chans[7] x n_frames
        # auger map shape: 
        # n_frames (0), n_subfames(1), n_y(2), n_x(3), n_chans(4)
    
        print(ke_map0.shape, ke_map0.sum())
        auger_ke0_imgs = np.transpose(self.auger_map, (4,0,1,2,3))[ke_map0,0,:,:]
        auger_ke1_imgs = np.transpose(self.auger_map, (4,0,1,2,3))[ke_map1,0,:,:]

        print(auger_ke0_imgs.shape, auger_ke1_imgs.shape)
        # Stored these arrays in object so could be updated/manipulated on demand more easily
        self.A = auger_ke0_imgs.mean(axis=0)
        self.B = auger_ke1_imgs.mean(axis=0)
        
        self.imview_auger.setImage(self.compute_image(self.A,self.B))
        
        self.lr0.setRegion((S['ke0_start'], S['ke0_stop']))
        self.lr1.setRegion((S['ke1_start'], S['ke1_stop']))
    
    def on_change_math_mode(self):
        self.imview_auger.setImage(self.compute_image(self.A,self.B))
    
    def on_change_roi(self):
        # Only need to update the spectrum if being calculated over ROI
        if self.settings['spectrum_over_ROI']:
            self.update_spectrum_display()
            
    def on_change_regions(self):
        S = self.settings
        S['ke0_start'], S['ke0_stop'] = self.lr0.getRegion()
        S['ke1_start'], S['ke1_stop'] = self.lr1.getRegion()
    
    def on_change_spectrum_over_ROI(self):
        self.update_spectrum_display()
    
    def on_change_subtract_spectrum_background(self):
        self.update_spectrum_display()
    
    def calculate_detector_efficiencies(self):
        # Step 1. Identify and extract data to compare
        
        # Determine highest, lowest, and middle energy detectors
        num_chans = self.ke.shape[0]
        
        det_rank = np.argsort(self.ke[:,0])
        det_low = det_rank[0]
        det_med = det_rank[len(det_rank)//2]
        det_high = det_rank[-1]
        
        # KE endpoints
        ke_lower = self.ke[det_high,0]
        ke_upper = self.ke[det_low,-1]
        
        ke_med_map = (self.ke[det_med,:] >= ke_lower) * (self.ke[det_med,:] <= ke_upper)
        
        # Extract data from the reference
        ke_med = self.ke[det_med, ke_med_map]
        auger_map_sum = np.sum(self.auger_map[:,0,:,:,0:7],axis=(1,2))
        data = np.transpose(auger_map_sum)
        data_med = data[det_med, ke_med_map]
        
        # Extract KE and data for other detector spectra 
        
        ke_rest = np.delete(self.ke, (det_med), axis = 0)
        data_rest = np.delete(data, (det_med), axis = 0)
        
        ke_step = self.ke[0,1] - self.ke[0,0]
        
        ke_map = (ke_rest > ke_med[0] - (ke_step-0.00001)) * (ke_rest < ke_med[-1] + (ke_step-0.00001))
        ke_sliced = ke_rest[ke_map]
        ke_sliced = ke_sliced.reshape((num_chans-1, len(ke_sliced)//(num_chans-1)))
        data_sliced = data_rest[ke_map].reshape((num_chans-1, ke_sliced.shape[1]))
        
        # Step 2. Interpolate
        data_intp = np.array([np.interp(ke_med, ke_sliced[idet], data_sliced[idet]) for idet in range(0, num_chans-1)])    
        # Add row back into stack
        data_join = np.insert(data_intp, det_med, data_med, axis=0)
        
        # Step 3. Integrate
        self.det_eff = np.sum(data_join, axis=1)/np.sum(data_join[det_med,:])
        
    def compute_image(self, A,B):
        mm = self.settings['math_mode']
        return np.transpose(eval(mm))
        
        
    def compute_total_spectrum0(self):
        from scipy import interpolate
        sum_Hz = self.auger_map[:,:,:,:,0].mean(axis=(1,2,3))
        x0 = self.ke[0,:]
        for i in range(1,7):
            x = self.ke[i,:]
            y=self.auger_map[:,:,:,:,i].mean(axis=(1,2,3))
            ff = interpolate.interp1d(x,y,bounds_error=False)
            sum_Hz += ff(x0)
        return sum_Hz/7.0

    def compute_total_spectrum(self, data = None):
        from scipy import interpolate
        total_spec = np.zeros(100, dtype=float)
        ke_interp = np.linspace(self.ke.min(), self.ke.max(), 100, dtype=float)
        for i in range(0,7):
            x = self.ke[i,:]
            if data==None:
                y = self.auger_map[:,:,:,:,i].mean(axis=(1,2,3))
            else:
                y = data[:,i]
            ff = interpolate.interp1d(x,y,bounds_error=False)
            total_spec += ff(ke_interp)
        return ke_interp, (total_spec/7.0)
    
    def subtract_background(self, x, y):
        # Fit a power law to the background
        x_min = self.settings['ke1_start']
        x_max = self.settings['ke1_stop']
        fit_map = (x > x_min) * (x < x_max)
        
        
        x_to_fit = x[fit_map]
        y_to_fit = y[fit_map]
        logx = np.log10(x_to_fit)
        logy = np.log10(y_to_fit)
        
        # define our (line) fitting function
        fitfunc = lambda p, x: p[0] + p[1] * x
        errfunc = lambda p, x, y: (y - fitfunc(p, x))
        
        # Calculate coefficient guess
        A = np.max(y_to_fit) * x_to_fit[np.argmax(y_to_fit)]
        pinit = [np.log10(A), -1.0]
        out = optimize.leastsq(errfunc, pinit,
                               args=(logx, logy), full_output=1)
        
        pfinal = out[0]
        index = pfinal[1]
        amp = 10.0**pfinal[0]
        fit_data = amp * x ** index
        
        return x, y-fit_data
    
    def update_spectrum_display(self):
        if self.settings['spectrum_over_ROI']:
#             roi_slice, roi_tr = self.poly_roi.getArraySlice(self.auger_map, self.im_auger, axes=(3,2))
#             print('ROI slice', roi_slice)
#             print('Local Positions', self.poly_roi.getLocalHandlePositions())
#             print('Scene Positions', self.poly_roi.getSceneHandlePositions())
#                    
            roi_auger_map = self.poly_roi.getArrayRegion(np.swapaxes(self.auger_map, 2, 3), self.im_auger, axes=(2,3))
            roi_auger_masked = np.ma.array(roi_auger_map, mask = roi_auger_map == 0)
            roi_auger_mean = roi_auger_masked.mean(axis=(1,2,3))
            
            #print(mapped_coords)
            if self.settings['subtract_spectrum_background']:
                plot_data = self.subtract_background(*self.compute_total_spectrum(data = roi_auger_mean))
                self.total_plotline.setData(*plot_data)
                for ii in range(7):
                    self.chan_plotlines[ii].setData([],[])
            else:
                self.total_plotline.setData(*self.compute_total_spectrum(data = roi_auger_mean))
                for ii in range(7):
                    self.chan_plotlines[ii].setData(self.ke[ii,:], roi_auger_mean[:,ii])                                                
            
        else:
            if self.settings['subtract_spectrum_background']:
                plot_data = self.subtract_background(*self.compute_total_spectrum())
                self.total_plotline.setData(*plot_data)
                for ii in range(7):
                    self.chan_plotlines[ii].setData([],[])
            else:
                self.total_plotline.setData(*self.compute_total_spectrum())
                for ii in range(7):
                    self.chan_plotlines[ii].setData(self.ke[ii,:],
                                                    self.auger_map[:,:,:,:,ii].mean(axis=(1,2,3)))
        

