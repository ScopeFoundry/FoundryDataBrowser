from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg
import pyqtgraph.dockarea as dockarea
from qtpy import QtWidgets, QtGui
from scipy import optimize
from scipy.ndimage.filters import gaussian_filter

import sys
#sys.path.insert(0, '/home/dbdurham/foundry_scope/FoundryDataBrowser/viewers')
from .drift_correction import register_translation_hybrid, shift_subpixel

class AugerSpecMapView(DataBrowserView):
    
    name = 'auger_spec_map'
    
    def setup(self):
        
        self.settings.New('drift_correct_type', dtype=str, initial='Pairwise', choices=('Pairwise','Pairwise + Running Avg'))
        
        self.settings.New('drift_correct_adc_chan', dtype=int)
        
        self.settings.New('drift_correct', dtype=bool)
        self.settings.get_lq('drift_correct').add_listener(self.on_change_drift_correct)
        
        self.settings.New('equalize_detectors', dtype=bool)
        self.settings.get_lq('equalize_detectors').add_listener(self.update_current_auger_map)
        
        self.settings.New('smooth_auger_sigma', dtype=float, vmin=0.0)
        self.settings.get_lq('smooth_auger_sigma').add_listener(self.update_current_auger_map)
        
        self.settings.New('ke0_start', dtype=float)
        self.settings.New('ke0_stop', dtype=float)
        self.settings.New('ke1_start', dtype=float)
        self.settings.New('ke1_stop', dtype=float)
        
        #Math mode now updates automatically on change
        self.settings.New('math_mode', dtype=str, initial='A')
        self.settings.get_lq('math_mode').add_listener(self.on_change_math_mode)
        
        self.settings.New('spectrum_over_ROI', dtype=bool)
        self.settings.get_lq('spectrum_over_ROI').add_listener(self.on_change_spectrum_over_ROI)
        
        # Subtract the B section (ke1_start through ke1_stop) by a power law fit
        self.settings.New('subtract_spectrum_background', dtype=bool)
        self.settings.get_lq('subtract_spectrum_background').add_listener(self.on_change_subtract_spectrum_background)
        
        self.settings.New('mean_spectrum_only', dtype=bool, initial=False)
        self.settings.get_lq('mean_spectrum_only').add_listener(self.on_change_mean_spectrum_only)
        
        for lqname in ['ke0_start', 'ke0_stop', 'ke1_start', 'ke1_stop']:
            self.settings.get_lq(lqname).add_listener(self.on_change_ke_settings)
        
        # Make plots on white background
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.ui = self.dockarea = dockarea.DockArea()
        
        self.dockarea.addDock(name='Settings', widget=self.settings.New_UI())
        
        # Spectrum plot
        self.graph_layout = pg.GraphicsLayoutWidget()        
        self.spec_plot = self.graph_layout.addPlot()
        self.legend = self.spec_plot.addLegend()
        self.spec_plot.setLabel('bottom','Electron Kinetic Energy')
        self.spec_plot.setLabel('left','Intensity (Hz)')
        #self.rect_plotdata = self.spec_plot.plot()
        #self.point_plotdata = self.spec_plot.plot(pen=(0,9))
        
        self.dockarea.addDock(name='Spec Plot', widget=self.graph_layout)

        self.lr0 = pg.LinearRegionItem(values=[0,1], brush=QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
        self.lr1 = pg.LinearRegionItem(values=[2,3], brush=QtGui.QBrush(QtGui.QColor(255, 0, 0, 50)))
        
        for lr in (self.lr0, self.lr1):
            lr.setZValue(10)
            self.spec_plot.addItem(lr, ignoreBounds=True)
            lr.sigRegionChangeFinished.connect(self.on_change_regions)
            
        self.chan_plotlines = []
        # define plotline color scheme going from orange -> yellow -> green
        R = np.linspace(220,0,4)
        G = np.linspace(220,100,4)
        
        plot_colors = [(R[0], G[0], 0), (R[1], G[0], 0), (R[0], G[1], 0), 
                       (R[2], G[0], 0), (R[0], G[2], 0), (R[3], G[0], 100),
                       (R[0], G[3], 0)]
        for ii in range(7):
            self.chan_plotlines.append(
                self.spec_plot.plot([0], pen=pg.mkPen(color=plot_colors[ii],width=2), name='chan ' + str(ii), width=20))
        self.total_plotline = self.spec_plot.plot(pen=pg.mkPen(color=(0,0,0), width=3), name='mean')
        
            
        # Images
        self.imview_sem0_stack = pg.ImageView()
        self.imview_sem0_stack.getView().invertY(False) # lower left origin
        self.imdockA_stack = self.dockarea.addDock(name='SE2 Image Stack', widget=self.imview_sem0_stack)
        
        self.imview_sem1_stack = pg.ImageView()
        self.imview_sem1_stack.getView().invertY(False) # lower left origin
        self.imdockB_stack = self.dockarea.addDock(name='InLens Image Stack', widget=self.imview_sem1_stack)
        
        self.imview_sem0 = pg.ImageView()
        self.imview_sem0.getView().invertY(False) # lower left origin
        self.imdockA = self.dockarea.addDock(name='SE2 Mean Image', widget=self.imview_sem0)

        self.imview_sem1 = pg.ImageView()
        self.imview_sem1.getView().invertY(False) # lower left origin
        self.imdockB = self.dockarea.addDock(name='InLens Mean Image', widget=self.imview_sem1)

        self.imview_auger = pg.ImageView()
        self.imview_auger.getView().invertY(False) # lower left origin
        self.imdockAuger = self.dockarea.addDock(name='Auger Map', widget=self.imview_auger)
        self.im_auger = self.imview_auger.getImageItem()
        
        # tab image and auger map docks
        self.dockarea.moveDock(self.imdockA_stack, 'above', self.imdockB_stack)
        self.dockarea.moveDock(self.imdockB, 'above', self.imdockA_stack)
        self.dockarea.moveDock(self.imdockA, 'above', self.imdockB)
        self.dockarea.moveDock(self.imdockAuger, 'above', self.imdockA)
        
        # Polygon ROI
        self.poly_roi = pg.PolyLineROI([[20,0], [20,20], [0,20]], pen = pg.mkPen((255,0,0),dash=[5,5], width=1.5), closed=True)
        #self.poly_roi = pg.RectROI([20, 20], [20, 20], pen=(0,9))
        #self.poly_roi = pg.CircleROI((0,0), (10,10) , movable=True, pen=(0,9))
        #self.poly_roi.addTranslateHandle((0.5,0.5))        
        self.imview_auger.getView().addItem(self.poly_roi)        
        self.poly_roi.sigRegionChanged[object].connect(self.on_change_roi)
        
        # Scalebar ROI
        self.scalebar = pg.LineROI([5, 5], [25, 5], width=0, pen = pg.mkPen(color=(255,255,0),width=4.5))
        self.imview_auger.getView().addItem(self.scalebar)
        
        # Create initial scalebar w/ text
        self.scale_text = pg.TextItem(color=(255,255,0), anchor=(0.5,1))
        self.imview_auger.getView().addItem(self.scale_text)
        self.scale_text.setFont(pg.QtGui.QFont('Arial', pointSize = 11))
        self.scalebar.sigRegionChanged[object].connect(self.on_change_scalebar)
        
        # Change handle colors so they don't appear by default, but do when hovered over
        scale_handles = self.scalebar.getHandles()
        scale_handles[0].currentPen.setColor(pg.mkColor(255,255,255,0))
        scale_handles[1].currentPen.setColor(pg.mkColor(255,255,255,0))
        # This disables the middle handle that allows line width change
        scale_handles[2].setOpacity(0.0)

    def is_file_supported(self, fname):
        return "auger_sync_raster_scan.h5" in fname

    def on_change_data_filename(self, fname=None):
        try:
            self.fname = fname
            print('opening hdf5 file...')
            self.dat = h5py.File(self.fname, 'r')
            print('hdf5 file loaded')
            self.H = self.dat['measurement/auger_sync_raster_scan/']
            h = self.h_settings = self.H['settings'].attrs
            print('copying arrays into memory...')
            print('adc map...')
            self.adc_map = np.array(self.H['adc_map'])
            self.settings.drift_correct_adc_chan.change_min_max(vmin=0, vmax=self.adc_map.shape[-1]-1)
            # ctr map is not very useful here...
#             print('ctr map...')
#             self.ctr_map = np.array(self.H['ctr_map'])
            print('auger map...')
            self.auger_map = np.array(self.H['auger_chan_map'], dtype=float)
            print('dataset arrays now available')
            time_per_px = self.auger_map[:,:,:,:,8:9]* 25e-9 # units of 25ns converted to seconds
            self.auger_map = self.auger_map[:,:,:,:,0:7]/time_per_px # auger map now in Hz
#             self.auger_sum_map = self.auger_map[:,:,:,:,0:7].mean(axis=4)
            
            print('loading ke...')
            self.ke = np.array(self.H['ke'])
            
            self.h_range = (h['h0'], h['h1'])
            self.v_range = (h['v0'], h['v1'])
            self.nPixels = (h['Nh'], h['Nv'])
            
            self.R = self.dat['hardware/sem_remcon/']
            r = self.r_settings = self.R['settings'].attrs
            self.full_size = r['full_size'] # in meters
            
            #scan_shape = self.adc_map.shape[:-1]
            
            # Close the h5 dataset, everything is stored in current memory now
            self.dat.close()
            
            # SEM image displays update
            print('setting SEM image stacks...')
            self.imview_sem0_stack.setImage(np.transpose(self.adc_map[:,:,:,:,0].mean(axis=1), (0,2,1)))
            self.imview_sem1_stack.setImage(np.transpose(self.adc_map[:,:,:,:,1].mean(axis=1), (0,2,1)))
            print('setting SEM mean image...')
            self.imview_sem0.setImage(np.transpose(self.adc_map[:,:,:,:,0].mean(axis=(0,1))))
            self.imview_sem1.setImage(np.transpose(self.adc_map[:,:,:,:,1].mean(axis=(0,1))))
            
            # Update Scale Bar
            self.on_change_scalebar()
            
            # Calculate relative detector efficiencies
            print('calculating detector efficiencies...')
            self.calculate_detector_efficiencies()
            
            # Auger map and display update
            print('updating auger map')
            self.update_current_auger_map()
            
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
            adc_chan = self.settings['drift_correct_adc_chan']
            num_frames = scan_shape[0]  # Consider allowing correcting only over a range in case data has blank or incorrect frames
            print('Correcting ' + str(num_frames) + ' frames...')
            print('adc map shape', self.adc_map.shape)
            print('auger map shape', self.auger_map.shape)
            
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
            
            if self.settings['drift_correct_type'] == 'Pairwise + Running Avg':
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
            else:
                self.adc_map = imstack
                self.auger_map = specstack
            
        else: #reload original images and auger from h5
            print('opening hdf5 file...')
            dat = h5py.File(self.fname, 'r')
            print('hdf5 file loaded')
            self.H = dat['measurement/auger_sync_raster_scan/']
            print('copying arrays into memory...')
            print('adc map...')
            self.adc_map = np.array(self.H['adc_map'])
            print('auger map...')
            self.auger_map = np.array(self.H['auger_chan_map'], dtype=float)
            print('dataset arrays now available')
            time_per_px = self.auger_map[:,:,:,:,8:9]* 25e-9 # units of 25ns converted to seconds
            self.auger_map = self.auger_map[:,:,:,:,0:7]/time_per_px # auger map now in Hz
            
            dat.close()
            
        # Display
        self.imview_sem0_stack.setImage(np.transpose(self.adc_map[:,:,:,:,0].mean(axis=1), (0,2,1)))
        self.imview_sem1_stack.setImage(np.transpose(self.adc_map[:,:,:,:,1].mean(axis=1), (0,2,1)))
        self.imview_sem0.setImage(np.transpose(self.adc_map[:,:,:,:,0].mean(axis=(0,1))))
        self.imview_sem1.setImage(np.transpose(self.adc_map[:,:,:,:,1].mean(axis=(0,1))))
        
        self.update_current_auger_map()
    
    def on_change_ke_settings(self):
        
        print ("on_change_ke_settings")
        S = self.settings
        print('ke shape', self.ke.shape)
        ke_map0 = (S['ke0_start'] < self.ke) * (self.ke < S['ke0_stop'])
        ke_map1 = (S['ke1_start'] < self.ke) * (self.ke < S['ke1_stop'])
        
        # KE of shape n_chans[7] x n_frames
        # auger map shape: 
        # n_frames (0), n_subfames(1), n_y(2), n_x(3), n_chans(4)
        
        if ke_map0.sum() == 0:
            self.A = np.zeros(self.current_auger_map.shape[2:4])
        else:
            auger_ke0_imgs = np.transpose(self.current_auger_map, (4,0,1,2,3))[ke_map0,0,:,:]
            self.A = auger_ke0_imgs.mean(axis=0)
        
        if ke_map1.sum() == 0:
            self.B = np.zeros(self.current_auger_map.shape[2:4])
        else:
            auger_ke1_imgs = np.transpose(self.current_auger_map, (4,0,1,2,3))[ke_map1,0,:,:]
            self.B = auger_ke1_imgs.mean(axis=0)

        # Stored these arrays in object so could be updated/manipulated on demand more easily
        
        self.imview_auger.setImage(self.compute_image(self.A,self.B))
        
        self.lr0.setRegion((S['ke0_start'], S['ke0_stop']))
        self.lr1.setRegion((S['ke1_start'], S['ke1_stop']))
    
    def on_change_math_mode(self):
        self.imview_auger.setImage(self.compute_image(self.A,self.B))
    
    def on_change_roi(self):
        # Only need to update the spectrum if being calculated over ROI
        if self.settings['spectrum_over_ROI']:
            self.update_spectrum_display()
    
    def on_change_scalebar(self):
        # Calculate the scale length (in pixels for now)
        scale_length = self.scalebar.size().x()
        # Calculate scale bar position and local midpoint
        scalebar_pos = self.scalebar.pos()
        midpoint_x = scalebar_pos.x() + 0.5 * scale_length * np.cos(np.deg2rad(self.scalebar.angle()))
        midpoint_y = scalebar_pos.y() + 0.5 * scale_length * np.sin(np.deg2rad(self.scalebar.angle()))
        
        # Convert from pixels to _______meters
        # 1. Determine pixel size in meters
        scan_size_h = ((self.h_range[1]-self.h_range[0])/20.0) * self.full_size
        scan_size_v = ((self.v_range[1]-self.v_range[0])/20.0) * self.full_size
        pixel_size_h = (scan_size_h/self.nPixels[0])
        pixel_size_v = (scan_size_v/self.nPixels[1])
        # 2. Convert scale lengths along x and y to meters
        scale_length_x = pixel_size_h * scale_length * np.cos(np.deg2rad(self.scalebar.angle()))
        scale_length_y = pixel_size_v * scale_length * np.sin(np.deg2rad(self.scalebar.angle()))
        # 3. Calculate new magnitude
        scale_length_m = np.sqrt(np.square(scale_length_x) + np.square(scale_length_y))
        
        if scale_length_m < 1e-6:
            scale_length_m *= 1e9
            scale_unit = ' nm'
        else:
            scale_length_m *= 1e6
            scale_unit = ' um'

        # Update scalebar text and position
        self.scale_text.setText(str(np.around(scale_length_m,decimals=1)) + scale_unit)
        self.scale_text.setPos(midpoint_x, midpoint_y)
        # FIX: setAngle doesn't seem to work on angles between 60 and 120 degrees? Bizarre...
        self.scale_text.setAngle(self.scalebar.angle())
           
    def on_change_regions(self):
        S = self.settings
        S['ke0_start'], S['ke0_stop'] = self.lr0.getRegion()
        S['ke1_start'], S['ke1_stop'] = self.lr1.getRegion()
    
    def on_change_spectrum_over_ROI(self):
        self.update_spectrum_display()
    
    def on_change_subtract_spectrum_background(self):
        self.update_spectrum_display()
        
    def on_change_mean_spectrum_only(self):
        print('mean_spectrum_only')
        for ii in range(7):
            self.chan_plotlines[ii].setVisible(not(self.settings['mean_spectrum_only']))
        self.legend.setVisible(not(self.settings['mean_spectrum_only']))
    
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
        sum_Hz = self.current_auger_map[:,:,:,:,0].mean(axis=(1,2,3))
        x0 = self.ke[0,:]
        for i in range(1,7):
            x = self.ke[i,:]
            y=self.current_auger_map[:,:,:,:,i].mean(axis=(1,2,3))
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
                y = self.current_auger_map[:,:,:,:,i].mean(axis=(1,2,3))
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
    
    def update_current_auger_map(self):
        # Initialize current auger map dataset
        self.current_auger_map = np.zeros(self.auger_map.shape)
        self.current_auger_map[:] = self.auger_map
        
        # Equalize detectors
        if self.settings['equalize_detectors']:
            self.current_auger_map /= self.det_eff
                
        # Smoothing
        sigma = self.settings['smooth_auger_sigma']
        if sigma > 0.0:
            self.current_auger_map = gaussian_filter(self.current_auger_map, (0,0,sigma,sigma,0))
        
        # Background subtraction
        
        # Update displays
        self.update_spectrum_display()
        self.on_change_ke_settings()
        
    
    def update_spectrum_display(self):
        if self.settings['spectrum_over_ROI']:
#             roi_slice, roi_tr = self.poly_roi.getArraySlice(self.auger_map, self.im_auger, axes=(3,2))
#             print('ROI slice', roi_slice)
#             print('Local Positions', self.poly_roi.getLocalHandlePositions())
#             print('Scene Positions', self.poly_roi.getSceneHandlePositions())
#                    
            roi_auger_map = self.poly_roi.getArrayRegion(np.swapaxes(self.current_auger_map, 2, 3), self.im_auger, axes=(2,3))
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
                                                    self.current_auger_map[:,:,:,:,ii].mean(axis=(1,2,3)))
        

