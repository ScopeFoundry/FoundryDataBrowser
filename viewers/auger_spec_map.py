from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg
import pyqtgraph.dockarea as dockarea
from qtpy import QtWidgets, QtGui
from scipy import optimize
from scipy.ndimage.filters import gaussian_filter, correlate1d
from scipy.signal import savgol_filter
from scipy import interpolate

import sys
import time
#sys.path.insert(0, '/home/dbdurham/foundry_scope/FoundryDataBrowser/viewers')
from .drift_correction import register_translation_hybrid, shift_subpixel, \
                              compute_pairwise_shifts, compute_retained_box, align_image_stack

class AugerSpecMapView(DataBrowserView):
    
    name = 'auger_spec_map'
    
    def setup(self):
        
        self.data_loaded = False
        
        self.settings.New('drift_correct_type', dtype=str, initial='Pairwise', choices=('Pairwise','Pairwise + Running Avg'))
        
        self.settings.New('drift_correct_adc_chan', dtype=int)
        
        self.settings.New('drift_correct', dtype=bool)
        self.settings.New('overwrite_alignment', dtype=bool)
        # if drift corrected datasets already exist, overwrite?
        
        self.settings.New('run_preprocess', dtype=bool)
        self.settings.get_lq('run_preprocess').add_listener(self.preprocess)
        
        self.settings.New('use_preprocess', dtype=bool, initial=True)
        self.settings.get_lq('use_preprocess').add_listener(self.load_new_data)
        
        self.settings.New('update_auger_map', dtype=bool, initial=False)
        self.settings.get_lq('update_auger_map').add_listener(self.update_current_auger_map)
        
        
        self.settings.New('equalize_detectors', dtype=bool)
        self.settings.New('normalize_by_pass_energy', dtype=bool)
        self.settings.New('spatial_smooth_sigma', dtype=float, vmin=0.0)        
        self.settings.New('spectral_smooth_type', dtype=str, choices=['None', 'Gaussian', 'Savitzky-Golay'], initial='None')
        self.settings.New('spectral_smooth_gauss_sigma', dtype=float, vmin=0.0)
        self.settings.New('spectral_smooth_savgol_width', dtype=int, vmin=0)
        self.settings.New('spectral_smooth_savgol_order', dtype=int, vmin=0, initial=2)
        
        # Assume same tougaard parameters everywhere
        self.settings.New('subtract_tougaard', dtype=bool)
        self.settings.New('R_loss', dtype=float)
        self.settings.New('E_loss', dtype=float)
        
        auger_lqs = ['equalize_detectors', 'normalize_by_pass_energy', 
                     'spatial_smooth_sigma','spectral_smooth_type',
                     'spectral_smooth_gauss_sigma','spectral_smooth_savgol_width',
                     'spectral_smooth_savgol_order', 'subtract_tougaard',
                     'R_loss', 'E_loss']
        
        # Link all the auger spectrum lqs to the update current auger map listener
        for alq in auger_lqs:
            self.settings.get_lq(alq).add_listener(self.update_current_auger_map)
        
        self.settings.New('ke0_start', dtype=float)
        self.settings.New('ke0_stop', dtype=float)
        self.settings.New('ke1_start', dtype=float)
        self.settings.New('ke1_stop', dtype=float)
        
        for lqname in ['ke0_start', 'ke0_stop', 'ke1_start', 'ke1_stop']:
            self.settings.get_lq(lqname).add_listener(self.on_change_ke_settings)
        
        # Subtract the B section (ke1_start through ke1_stop) by a power law fit
        self.settings.New('subtract_ke1', dtype=str, choices=['None','Linear','Power Law'], initial='None')
        self.settings.get_lq('subtract_ke1').add_listener(self.update_current_auger_map)
        
        #Math mode now updates automatically on change
        self.settings.New('math_mode', dtype=str, initial='A')
        self.settings.get_lq('math_mode').add_listener(self.on_change_math_mode)
        
        self.settings.New('AB_mode', dtype=str, choices=['Mean', 'Integral'], initial='Mean')
        self.settings.get_lq('AB_mode').add_listener(self.on_change_ke_settings)
        
        self.settings.New('spectrum_over_ROI', dtype=bool)
        self.settings.get_lq('spectrum_over_ROI').add_listener(self.on_change_spectrum_over_ROI)
        
        self.settings.New('analysis_over_spectrum', dtype=bool)
        self.settings.get_lq('analysis_over_spectrum').add_listener(self.update_current_auger_map)
        
        self.settings.New('mean_spectrum_only', dtype=bool, initial=False)
        self.settings.get_lq('mean_spectrum_only').add_listener(self.on_change_mean_spectrum_only)
        
        # Make plots on white background
        #pg.setConfigOption('background', 'w')
        #pg.setConfigOption('foreground', 'k')
        
        self.ui = self.dockarea = dockarea.DockArea()
        
        # List of settings to include in preprocessing tab
        names_prep = ['drift_correct_type','drift_correct_adc_chan', 'drift_correct',
                      'overwrite_alignment', 'run_preprocess']
        
        self.setdock = self.dockarea.addDock(name='Settings', position='left', 
                              widget=self.settings.New_UI(exclude=names_prep))
        self.prepdock = self.dockarea.addDock(name='Preprocess', position='left',
                              widget=self.settings.New_UI(include=names_prep))
        self.dockarea.moveDock(self.setdock, 'above', self.prepdock)
        
            
        # Images
        self.imview_sem0_stack = pg.ImageView()
        self.imview_sem0_stack.getView().invertY(False) # lower left origin
        self.imdockA_stack = self.dockarea.addDock(name='SE2 Image Stack', widget=self.imview_sem0_stack)
        
        self.imview_sem1_stack = pg.ImageView()
        self.imview_sem1_stack.getView().invertY(False) # lower left origin
        self.imdockB_stack = self.dockarea.addDock(name='InLens Image Stack', position='right', widget=self.imview_sem1_stack)
        
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
        
        # Spectrum plot
        self.graph_layout = pg.GraphicsLayoutWidget()        
        self.spec_plot = self.graph_layout.addPlot()
        self.legend = self.spec_plot.addLegend()
        self.spec_plot.setLabel('bottom','Electron Kinetic Energy')
        self.spec_plot.setLabel('left','Intensity (Hz)')
        #self.rect_plotdata = self.spec_plot.plot()
        #self.point_plotdata = self.spec_plot.plot(pen=(0,9))
        
        self.dockarea.addDock(name='Spec Plot',position='bottom', widget=self.graph_layout)

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

    def is_file_supported(self, fname):
        return "auger_sync_raster_scan.h5" in fname

    def on_change_data_filename(self, fname=None):
        if fname == "0":
            return
        try:
            # FIX: Should close the h5 file that was previously open
            # FIX: h5 file should also be closed on program close
            self.data_loaded = False
            self.fname = fname
            print('opening hdf5 file...')
            self.dat = h5py.File(self.fname, 'r+')
            print('hdf5 file loaded')
            self.H = self.dat['measurement/auger_sync_raster_scan/']
            h = self.h_settings = dict(self.H['settings'].attrs)
            
            self.h_range = (h['h0'], h['h1'])
            self.v_range = (h['v0'], h['v1'])
            self.nPixels = (h['Nh'], h['Nv'])
            
            self.R = self.dat['hardware/sem_remcon/']
            r = self.r_settings = self.R['settings'].attrs
            self.full_size = r['full_size'] # in meters
            
            #scan_shape = self.adc_map.shape[:-1]
            
#             # Close the h5 dataset, everything is stored in current memory now
#             self.dat.close()
            self.data_loaded = True
            self.load_new_data()
            
            #self.update_display()
        except Exception as err:
            print(err)
            self.imview_auger.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("Failed to load %s: %s" %(fname, err))
            raise(err)
    
    def preprocess(self):
        if not self.data_loaded:
            return
        if self.settings['drift_correct']:
            # Need to read existing datasets here in case of changes since loading file initially
            h_datasets = list(self.H.keys())
            if self.settings['overwrite_alignment'] or not('auger_chan_map_aligned' in h_datasets):
                if 'auger_chan_map_aligned' in h_datasets:
                    del self.H['adc_map_aligned']
                    del self.H['auger_chan_map_aligned']
                t0 = time.time()
                self.drift_correct()
                tdc = time.time()
                print('Drift correct time', tdc-t0)
                # refer to aligned data
                self.adc_map_h5 = self.adc_map_aligned_h5
                self.auger_map_h5 = self.auger_map_aligned_h5
        
        else:
            # refer to raw data 
            self.adc_map_h5 = self.adc_map_raw_h5
            self.auger_map_h5 = self.auger_map_raw_h5
        
        t0 = time.time()
        # Update displays
        self.imview_sem0_stack.setImage(np.transpose(self.adc_map_h5[:,:,:,:,0].mean(axis=1), (0,2,1)))
        self.imview_sem1_stack.setImage(np.transpose(self.adc_map_h5[:,:,:,:,1].mean(axis=1), (0,2,1)))
        self.imview_sem0.setImage(np.transpose(self.adc_map_h5[:,:,:,:,0].mean(axis=(0,1))))
        self.imview_sem1.setImage(np.transpose(self.adc_map_h5[:,:,:,:,1].mean(axis=(0,1))))
        tsup = time.time()
        print('update display time', tsup-t0)
        
        # Update analysis
        self.update_current_auger_map()
        tcam = time.time()
        print('update auger map time', tcam-tsup)
        
    def drift_correct(self):
        
        t0 = time.time()
        correct_chan = self.settings['drift_correct_adc_chan']
        shift = compute_pairwise_shifts(self.adc_map_h5[:,0,:,:,correct_chan])
        tps = time.time()
        print('pairwise shifts time', tps-t0)
        shift = np.concatenate([np.zeros((2,1)), shift],axis=1)
        # Cumulative sum defines shift with respect to original image for each image
        # Maxima and minima in cumulative x and y shifts defines box within which all images have defined pixels
        shift_cumul = np.cumsum(shift, axis=1)
        
        scan_shape_adc = self.adc_map_h5.shape
        scan_shape_auger = self.auger_map_h5.shape
        boxfd, boxdims = compute_retained_box(shift_cumul, 
                                              (scan_shape_adc[2], scan_shape_adc[3]))
        trb = time.time()
        print('retained box time', trb-tps)
        align_shape_adc = (scan_shape_adc[0], scan_shape_adc[1], boxdims[0], boxdims[1], scan_shape_adc[4])
        align_shape_auger = (scan_shape_auger[0], scan_shape_auger[1], boxdims[0], boxdims[1], scan_shape_auger[4])
        self.adc_map_aligned_h5 = self.H.create_dataset('adc_map_aligned', 
                                                        align_shape_adc, 
                                                        self.adc_map_h5.dtype)
        self.auger_map_aligned_h5 = self.H.create_dataset('auger_chan_map_aligned', 
                                                          align_shape_auger, 
                                                          self.auger_map_h5.dtype)
        tdat = time.time()
        print('create dataset time', tdat-trb)
        # Shift images to align
        for iFrame in range(0, scan_shape_adc[0]):
            # Shift adc map
            for iDet in range(0, align_shape_adc[-1]):
                adc_shift = shift_subpixel(self.adc_map_h5[iFrame,0,:,:,iDet], 
                                           dx=shift_cumul[1, iFrame], 
                                           dy=shift_cumul[0, iFrame])
                self.adc_map_aligned_h5[iFrame,0,:,:,iDet] = np.real(adc_shift[boxfd[0]:boxfd[1], boxfd[2]:boxfd[3]])
            # Shift spectral data
            for iDet in range(0, align_shape_auger[-1]):
                auger_shift = shift_subpixel(self.auger_map_h5[iFrame,0,:,:,iDet], 
                                             dx=shift_cumul[1, iFrame], 
                                             dy=shift_cumul[0, iFrame])
                self.auger_map_aligned_h5[iFrame,0,:,:,iDet] = np.real(auger_shift[boxfd[0]:boxfd[1], boxfd[2]:boxfd[3]])
        talign = time.time()
        print('align datasets time', talign-tdat)
        
#         if self.settings['drift_correct_type'] == 'Pairwise + Running Avg':
#             #### Phase 2: Running Average ####
#             
#             # Update the image shape
#             imshape = imstack.shape
#             imstack_run = imstack.copy()
#             specstack_run = specstack.copy()
#             
#             # Prepare window function (Hann)
#             win = np.outer(np.hanning(imshape[2]),np.hanning(imshape[3]))
#             
#             # Shifts to running average
#             shift = np.zeros((2, num_frames))
#             image = imstack[0,0,:,:,adc_chan]
#             for iFrame in range(1, num_frames):
#                 offset_image = imstack[iFrame,0,:,:,adc_chan]
#                 # Calculate shift
#                 shift[:,iFrame], error, diffphase = register_translation_hybrid(image*win, offset_image*win, exponent = 0.3, upsample_factor = 100)
#                 # Perform shifts
#                 # Shift adc map
#                 for iDet in range(0, imstack.shape[4]):
#                     imstack_run[iFrame,0,:,:,iDet] = shift_subpixel(imstack[iFrame,0,:,:,iDet], dx = shift[1,iFrame], dy = shift[0, iFrame])
#                  # Shift spectral data
#                 for iDet in range(0, specstack.shape[4]):
#                     specstack_run[iFrame,0,:,:,iDet] = shift_subpixel(specstack[iFrame,0,:,:,iDet], dx = shift[1,iFrame], dy = shift[0, iFrame])
#                 # Update running average
#                 image = (iFrame/(iFrame+1)) * image + (1/(iFrame+1)) * imstack_run[iFrame,0,:,:,adc_chan]
#             # Shifts are defined as [y, x] where y is shift of imaging location with respect to positive y axis, similarly for x
#             
#             # Determining coordinates of fully defined box for original image
# 
#             shift_y = shift[0,:]
#             shift_x = shift[1,:]
#             
#             # NOTE: scan_shape indices 2, 3 correspond to y, x
#             y1 = int(round(np.max(shift_y[shift_y >= 0])+0.001, 0))
#             y2 = int(round(imshape[2] + np.min(shift_y[shift_y <= 0])-0.001, 0))
#             x1 = int(round(np.max(shift_x[shift_x >= 0])+0.001, 0))
#             x2 = int(round(imshape[3] + np.min(shift_x[shift_x <= 0])-0.001, 0))
#             
#             boxfd = np.array([y1, y2, x1, x2])
#             boxdims = (boxfd[1]-boxfd[0], boxfd[3]-boxfd[2])
#             
#             # Keep only preserved data
#             self.adc_map = np.real(imstack_run[:,:,boxfd[0]:boxfd[1], boxfd[2]:boxfd[3],:])
#             self.auger_map = np.real(specstack_run[:,:,boxfd[0]:boxfd[1], boxfd[2]:boxfd[3],:])
#         else:
#             self.adc_map = imstack
#             self.auger_map = specstack
            
    def load_new_data(self):
        if not self.data_loaded:
            return
        
        if self.settings['use_preprocess']:
            h_datasets = list(self.H.keys())
            if 'adc_map_prep' in h_datasets:
                self.adc_map_h5 = self.H['adc_map_prep']
            if 'auger_map_prep' in h_datasets:
                self.auger_map_h5 = self.H['auger_map_prep']
            if 'ke_prep' in h_datasets:
                self.ke = self.H['ke_prep'][:]
        else:
            self.adc_map_h5 = self.H['adc_map'] 
            self.auger_map_h5 = self.H['auger_chan_map']
            self.ke = self.H['ke'][:]
            # Calculate relative detector efficiencies
            print('calculating detector efficiencies...')
            self.calculate_detector_efficiencies()
        
        # SEM image displays update
        # FIX: Using auger stack for now to check correctness
        print('setting SEM image stacks...')
        self.imview_sem0_stack.setImage(np.transpose(self.auger_map_h5[:,:,:,:,0].mean(axis=1), (0,2,1)))
        self.imview_sem1_stack.setImage(np.transpose(self.adc_map_h5[:,:,:,:,1].mean(axis=1), (0,2,1)))
        print('setting SEM mean image...')
        self.imview_sem0.setImage(np.transpose(self.auger_map_h5[:,:,:,:,0].mean(axis=(0,1))))
        self.imview_sem1.setImage(np.transpose(self.adc_map_h5[:,:,:,:,1].mean(axis=(0,1))))
        
        # Update Scale Bar
        self.on_change_scalebar()
        
        # Auger map and display update
        print('updating auger map')
        self.update_current_auger_map()
    
    def on_change_ke_settings(self):
        if not self.data_loaded:
            return
        
        print ("on_change_ke_settings")
        S = self.settings
        print('ke shape', self.ke.shape)
        ke_map0 = (S['ke0_start'] < self.ke) * (self.ke < S['ke0_stop'])
        ke_map1 = (S['ke1_start'] < self.ke) * (self.ke < S['ke1_stop'])
        
        # KE of shape n_chans[7] x n_frames
        # auger map shape: 
        # n_frames (0), n_subfames(1), n_y(2), n_x(3), n_chans(4)
        
        
        # FIX: integral assumes all measured points count evenly towards integral, 
        # but depending on dispersion they may be clustered which calls into question
        # the validity of this "integral" approximation
        if ke_map0.sum() == 0:
            self.A = np.zeros(self.current_auger_map.shape[2:4])
        else:
            auger_ke0_imgs = np.transpose(self.current_auger_map, (4,0,1,2,3))[ke_map0,0,:,:]
            if self.settings['AB_mode'] == 'Mean':
                self.A = auger_ke0_imgs.mean(axis=0)
            elif self.settings['AB_mode'] == 'Integral':
                deltaE = (S['ke0_stop'] - S['ke0_start'])/ke_map0.sum()
                self.A = auger_ke0_imgs.sum(axis=0) * deltaE
        
        if ke_map1.sum() == 0:
            self.B = np.zeros(self.current_auger_map.shape[2:4])
        else:
            auger_ke1_imgs = np.transpose(self.current_auger_map, (4,0,1,2,3))[ke_map1,0,:,:]
            if self.settings['AB_mode'] == 'Mean':
                self.B = auger_ke1_imgs.mean(axis=0)
            elif self.settings['AB_mode'] == 'Integral':
                deltaE = (S['ke0_stop'] - S['ke0_start'])/ke_map1.sum()
                self.B = auger_ke1_imgs.sum(axis=0) * deltaE

        # Stored these arrays in object so could be updated/manipulated on demand more easily
        
        self.imview_auger.setImage(self.compute_image(self.A,self.B))
        
        self.lr0.setRegion((S['ke0_start'], S['ke0_stop']))
        self.lr1.setRegion((S['ke1_start'], S['ke1_stop']))
    
    def on_change_math_mode(self):
        if not self.data_loaded:
            return        
        self.imview_auger.setImage(self.compute_image(self.A,self.B))
    
    def on_change_roi(self):
        if not self.data_loaded:
            return
        
        # Only need to update the spectrum if being calculated over ROI
        if self.settings['spectrum_over_ROI']:
            self.update_spectrum_display()
    
    def on_change_scalebar(self):
        if not self.data_loaded:
            return
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
        if not self.data_loaded:
            return
        
        S = self.settings
        S['ke0_start'], S['ke0_stop'] = self.lr0.getRegion()
        S['ke1_start'], S['ke1_stop'] = self.lr1.getRegion()
    
    def on_change_spectrum_over_ROI(self):
        self.update_spectrum_display()
    
    def on_change_subtract_ke1_powerlaw(self):
        self.update_spectrum_display()
        
    def on_change_mean_spectrum_only(self):
        if not self.data_loaded:
            return
        print('mean_spectrum_only')
        for ii in range(7):
            self.chan_plotlines[ii].setVisible(not(self.settings['mean_spectrum_only']))
        self.legend.setVisible(not(self.settings['mean_spectrum_only']))
    
    def calculate_detector_efficiencies(self):
        if not self.data_loaded:
            return
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
        auger_map_sum = np.sum(self.auger_map_h5[:,0,:,:,0:7],axis=(1,2))
        data = np.transpose(auger_map_sum)
        data_med = data[det_med, ke_med_map]
        
        # Extract KE and data for other detector spectra 
        
        ke_rest = np.delete(self.ke, (det_med), axis = 0)
        data_rest = np.delete(data, (det_med), axis = 0)
        
        ke_step = self.ke[0,1] - self.ke[0,0]
        
        ke_map = (ke_rest > ke_med[0] - 0.00001) * (ke_rest < ke_med[-1] + (ke_step-0.00001))
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
        if not self.data_loaded:
            return        
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

    def compute_total_spectrum(self, data = np.array([])):
        from scipy import interpolate
        n_frames = self.ke.shape[1]
        self.total_spec = np.zeros(n_frames, dtype=float)
        self.ke_interp = np.linspace(self.ke.min(), self.ke.max(), n_frames, dtype=float)
        num_chans = self.current_auger_map.shape[-1]
        for i in range(0,num_chans):
            x = self.ke[i,:]
            if data.size==0:
                y = self.current_auger_map[:,:,:,:,i].mean(axis=(1,2,3))
            else:
                y = data[:,i]
            ff = interpolate.interp1d(x,y,bounds_error=False)
            self.total_spec += ff(self.ke_interp)
    
    def update_current_auger_map(self):
        
        if self.settings['update_auger_map']:
            # Initialize copy of auger map in memory (current_auger_map)
            # this copy will be modified throughout the analysis (needs to be float to handle the math)
            auger_shape = self.auger_map_h5.shape
            if auger_shape[-1] == 10:
                self.current_auger_map = np.array(self.auger_map_h5[:,:,:,:,0:7], dtype='float')
                # Convert to Hz
                time_per_px = self.auger_map_h5[:,:,:,:,8:9]* 25e-9 # units of 25ns converted to seconds
                self.current_auger_map /= time_per_px # auger map now in Hz
            else:
                self.current_auger_map = np.array(self.auger_map_h5, dtype='float')
                # FIX: Auger map currently in counts since detectors have been summed but
                # time channel was not stored
            
            # Equalize detectors (does not apply to preprocess since detector averaging is already done)
            if self.settings['equalize_detectors'] and not(self.settings['use_preprocess']):
                self.current_auger_map /= self.det_eff
            
            #normalize counts by spec resolution, Hz/eV
            
            if self.settings['normalize_by_pass_energy']:
                self.spec_plot.setLabel('left','Intensity (Hz/eV)')
                spec_dispersion = 0.02  #Omicron SCA per-channel resolution/pass energy
                if self.h_settings['CAE_mode']:
                    self.current_auger_map /= spec_dispersion * self.h_settings['pass_energy']
                else:
                    #in CRR mode pass energy is KE / crr_ratio
                    self.current_auger_map *= self.h_settings['crr_ratio'] / (spec_dispersion * self.ke)
            else:
                self.spec_plot.setLabel('left','Intensity (Hz)')
            
            # Spatial smoothing before analysis in either case
            sigma_xy = self.settings['spatial_smooth_sigma'] # In terms of px?
            if sigma_xy > 0.0:
                print('spatial smoothing...')
                self.current_auger_map = gaussian_filter(self.current_auger_map, (0,0,sigma_xy,sigma_xy,0))
            
            if not self.settings['analysis_over_spectrum']:
                self.perform_map_analysis()
            
            # Update displays
            self.update_spectrum_display()
            self.on_change_ke_settings() 
    
    def update_spectrum_display(self):
        if not self.data_loaded:
            return
        
        # Calculate the average spectrum over the image OR the ROI
        if self.settings['spectrum_over_ROI']:
            roi_auger_map = self.poly_roi.getArrayRegion(np.swapaxes(self.current_auger_map, 2, 3), self.im_auger, axes=(2,3))
            roi_auger_masked = np.ma.array(roi_auger_map, mask = roi_auger_map == 0)
            space_avg_spectra = roi_auger_masked.mean(axis=(1,2,3))
        else:
            space_avg_spectra = self.current_auger_map.mean(axis=(1,2,3))
        
        
#             roi_slice, roi_tr = self.poly_roi.getArraySlice(self.auger_map, self.im_auger, axes=(3,2))
#             print('ROI slice', roi_slice)
#             print('Local Positions', self.poly_roi.getLocalHandlePositions())
#             print('Scene Positions', self.poly_roi.getSceneHandlePositions())
#                    
            #print(mapped_coords)
            
        # compute and condition total spectrum
        self.compute_total_spectrum(data = space_avg_spectra)
        if self.settings['analysis_over_spectrum']:
            self.perform_spectral_analysis()
        
        # Display all spectra
        num_chans = self.current_auger_map.shape[-1]
        self.total_plotline.setData(self.ke_interp, self.total_spec)
        for ii in range(num_chans):
            self.chan_plotlines[ii].setData(self.ke[ii,:], space_avg_spectra[:,ii])
            self.chan_plotlines[ii].setVisible(True)
        if num_chans < 7:
            for jj in range(num_chans,7):
                self.chan_plotlines[jj].setVisible(False)
    
    def perform_map_analysis(self):
        # Smoothing (presently performed for individual detectors)
        # FIX: MAY NEED A SUM MAP THAT ALIGNS DETECTOR CHANNELS AND SUMS
        sigma_spec = self.settings['spectral_smooth_gauss_sigma'] # In terms of frames?
        width_spec = self.settings['spectral_smooth_savgol_width'] # In terms of frames?
        order_spec = self.settings['spectral_smooth_savgol_order']
            
        if self.settings['spectral_smooth_type'] == 'Gaussian':
            print('spectral smoothing...')
            self.current_auger_map = gaussian_filter(self.current_auger_map, (sigma_spec,0,0,0,0))
        elif self.settings['spectral_smooth_type'] == 'Savitzky-Golay':
            # Currently always uses 4th order polynomial to fit
            print('spectral smoothing...')
            self.current_auger_map = savgol_filter(self.current_auger_map, 1 + 2*width_spec, order_spec, axis=0)
 
        # Background subtraction (implemented detector-wise currently)
        # NOTE: INSUFFICIENT SPATIAL SMOOTHING MAY GIVE INACCURATE OR EVEN INF RESULTS
        if not(self.settings['subtract_ke1'] == 'None'):
            print('Performing background subtraction...')
            for iDet in range(self.current_auger_map.shape[-1]):
                # Fit a power law to the background
                # get background range
                ke_min = self.settings['ke1_start']
                ke_max = self.settings['ke1_stop']
                fit_map = (self.ke[iDet] > ke_min) * (self.ke[iDet] < ke_max)
                ke_to_fit = self.ke[iDet,fit_map]
                spec_to_fit = self.current_auger_map[fit_map,0,:,:,iDet].transpose(1,2,0)
                
                if self.settings['subtract_ke1'] == 'Power Law':
                    # Fit power law
                    A, m = self.fit_powerlaw(ke_to_fit, spec_to_fit)
                    ke_mat = np.tile(self.ke[iDet], (spec_to_fit.shape[0],spec_to_fit.shape[1],1)).transpose(2,0,1)
                    A = np.tile(A, (self.ke.shape[1], 1, 1))
                    m = np.tile(m, (self.ke.shape[1], 1, 1))
                    bg = A * ke_mat**m
                elif self.settings['subtract_ke1'] == 'Linear':
                    # Fit line
                    m, b = self.fit_line(ke_to_fit, spec_to_fit)
                    ke_mat = np.tile(self.ke[iDet], (spec_to_fit.shape[0],spec_to_fit.shape[1],1)).transpose(2,0,1)
                    m = np.tile(m, (self.ke.shape[1], 1, 1))
                    b = np.tile(b, (self.ke.shape[1], 1, 1))
                    bg = m * ke_mat + b
                
                self.current_auger_map[:,0,:,:,iDet] -= bg
        
        if self.settings['subtract_tougaard']:
            R_loss = self.settings['R_loss']
            E_loss = self.settings['E_loss']
            dE = self.ke[0,1] - self.ke[0,0]
            # Always use a kernel out to 3 * E_loss to ensure enough feature size
            ke_kernel = np.arange(0, 3*E_loss, abs(dE))
            if not np.mod(len(ke_kernel),2) == 0:
                ke_kernel = np.arange(0, 3*E_loss+dE, abs(dE))
            self.K_toug = (8.0/np.pi**2)*R_loss*E_loss**2 * ke_kernel / ((2.0*E_loss/np.pi)**2 + ke_kernel**2)**2
            # Normalize the kernel so the its area is equal to R_loss
            self.K_toug /= (np.sum(self.K_toug) * dE)/R_loss
            self.current_auger_map -= dE * correlate1d(self.current_auger_map, self.K_toug,
                                                        mode='nearest', origin=-len(ke_kernel)//2, axis=0)
    
    def perform_spectral_analysis(self):
        # FIX: Consolidate with map analysis
        # Performs same analysis functions as the map, but just on the single (1D) total spectrum
        
        # Smoothing (presently performed for individual detectors)
        sigma_spec = self.settings['spectral_smooth_gauss_sigma'] # In terms of frames?
        width_spec = self.settings['spectral_smooth_savgol_width'] # In terms of frames?
        order_spec = self.settings['spectral_smooth_savgol_order']
            
        if self.settings['spectral_smooth_type'] == 'Gaussian':
            print('spectral smoothing...')
            self.total_spec = gaussian_filter(self.total_spec, sigma_spec)
        elif self.settings['spectral_smooth_type'] == 'Savitzky-Golay':
            # Currently always uses 4th order polynomial to fit
            print('spectral smoothing...')
            self.total_spec = savgol_filter(self.total_spec, 1 + 2*width_spec, order_spec)
 
        # Background subtraction (implemented detector-wise currently)
        # NOTE: INSUFFICIENT SPATIAL SMOOTHING MAY GIVE INACCURATE OR EVEN INF RESULTS
        if not(self.settings['subtract_ke1'] == 'None'):
            print('Performing background subtraction...')
            # Fit a power law to the background
            # get background range
            ke_min = self.settings['ke1_start']
            ke_max = self.settings['ke1_stop']
            fit_map = (self.ke_interp > ke_min) * (self.ke_interp < ke_max)
            ke_to_fit = self.ke_interp[fit_map]
            spec_to_fit = self.total_spec[fit_map]
            
            if self.settings['subtract_ke1'] == 'Power Law':
                # Fit power law
                A, m = self.fit_powerlaw(ke_to_fit, spec_to_fit)
                bg = A * self.ke_interp**m
            elif self.settings['subtract_ke1'] == 'Linear':
                # Fit line (there may be an easier way for 1D case)
                m, b = self.fit_line(ke_to_fit, spec_to_fit)
                bg = m * self.ke_interp + b
            
            self.total_spec -= bg
        
        if self.settings['subtract_tougaard']:
            R_loss = self.settings['R_loss']
            E_loss = self.settings['E_loss']
            dE = self.ke_interp[1] - self.ke_interp[0]
            # Always use a kernel out to 3 * E_loss to ensure enough feature size
            ke_kernel = np.arange(0, 3*E_loss, abs(dE))
            if not np.mod(len(ke_kernel),2) == 0:
                ke_kernel = np.arange(0, 3*E_loss+dE, abs(dE))
            self.K_toug = (8.0/np.pi**2)*R_loss*E_loss**2 * ke_kernel / ((2.0*E_loss/np.pi)**2 + ke_kernel**2)**2
            # Normalize the kernel so the its area is equal to R_loss
            self.K_toug /= (np.sum(self.K_toug) * dE)/R_loss
            self.total_spec -= dE * correlate1d(self.total_spec, self.K_toug,
                                                        mode='nearest', origin=-len(ke_kernel)//2, axis=0)


    def fit_powerlaw(self, x, y):
        # Takes x data (1d array) and y data (Nd array)
        # last dimension of y array is interpolation dimension
        # Returns coefficients A,m for the powerlaw y = Ax^m that provide least squares best fit
        # Solved algebraically for speed
        # NOTE: This form goes to zero in correct limits for either primary or secondary electron backgrounds, but not combined
        # e.g. secondary background has m < 0 such that y -> 0 as x -> inf
        # e.g. primary background ahs m > 0 such that y -> 0 as x -> 0 
        
        # First, must interpolate data to equally spaced points in log space
        # Currently, interpolation is linear
        
        f_interp = interpolate.interp1d(x, y)
        x_lsp = 10**np.linspace(np.log10(x[0]), np.log10(x[-1]), len(x)) # log equally spaced x
        x_lsp[0] = x[0] # This prevents 10^ log10 operation from moving x values slightly out of interpolation range
        x_lsp[-1] = x[-1]
        y_lsp = f_interp(x_lsp)
        
        # Then, solve linear least squares matrix equations for A, m
        # XB = y where X is a matrix based on x data, B is (log(A),m), and y is the y data
        # In this case, equation is generated using y_i = 1*(log(A)) + x_i * m where x and y are log(x) and log(y)
        
        # Generate X matrix (X_i = [1, x_i])
        X = np.ones((len(x_lsp), 2))
        X[:,1] = np.log10(x_lsp[:])
        
        # Solve "normal equations" for B that minimizes least sq
        # B = C*y = {((X^T)X)^(-1) * (X^T)} y
        
        C = np.linalg.inv(X.T.dot(X)).dot(X.T)
        if len(y.shape) < 2:
            B = C.dot(np.log10(y_lsp))
        else:
            B = C.dot(np.log10(np.swapaxes(y_lsp, -1, 1)))
        return 10**B[0], B[1]
    
    def fit_line(self, x, y):
        # Takes x data (1d array) and y data (Nd array)
        # last dimension of y array is fit dimension
        # Solve linear least squares matrix equations for m, b of y = mx + b
        # Returns m, b
        # y_i = 1*b + x_i * m
        
        # Generate X matrix (X_i = [1, x_i])
        X = np.ones((len(x), 2))
        X[:,1] = x[:]
        
        # Solve "normal equations" for B that minimizes least sq
        # B = C*y = {((X^T)X)^(-1) * (X^T)} y
        
        C = np.linalg.inv(X.T.dot(X)).dot(X.T)
        if len(y.shape) < 2:
            B = C.dot(y)
        else:
            B = C.dot(np.swapaxes(y, -1, 1))
        return B[1], B[0]
        
