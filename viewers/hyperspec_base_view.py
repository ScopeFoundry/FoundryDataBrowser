'''
Created on Aug 7, 2019

@author: lab
'''

from ScopeFoundry.data_browser import DataBrowserView
from FoundryDataBrowser.viewers.plot_n_fit import PlotNFit
from FoundryDataBrowser.viewers.scalebars import ConfocalScaleBar
from ScopeFoundry.widgets import RegionSlicer
from ScopeFoundry.helper_funcs import sibling_path
from ScopeFoundry.logged_quantity import LQCollection

from scipy.stats import spearmanr

import os

import time
from datetime import datetime
import h5py

import numpy as np

from qtpy import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.dockarea as dockarea
from lxml import includes

class HyperSpectralBaseView(DataBrowserView):
    
    name = 'HyperSpectralBaseView'
    
    def setup(self):

        ## Dummy data Structures (override in func:self.load_data())
        self.hyperspec_data = np.arange(10*10*34).reshape( (10,10,34) )
        self.display_image = self.hyperspec_data.sum(-1)# np.random.rand(10,10)
        self.spec_x_array = np.arange(34)

        # Call :func:set_scalebar_params() during self.load_data() to add a scalebar!
        self.scalebar_type = None


        # Will be filled derived maps and x_arrays
        self.display_images = dict()
        self.spec_x_arrays = dict()   

        
        ## Graphs and Interface 
        self.line_colors = ['w', 'r', 'b', 'y', 'm', 'c', 'g']
        self.plot_n_fit = PlotNFit(Ndata_lines=2, pens=['g']+self.line_colors)


        # Docks
        self.ui = self.dockarea = dockarea.DockArea()
        self.image_dock = self.dockarea.addDock(name='Image')
        self.spec_dock = self.dockarea.addDock(self.plot_n_fit.graph_dock)
        self.settings_dock = self.dockarea.addDock(name='settings', 
                                                   position='left', relativeTo=self.image_dock)
        self.export_dock = self.dockarea.addDock(name='export & adv. settings', 
                                                 position='below', relativeTo=self.settings_dock)  
        self.dockarea.addDock(self.plot_n_fit.settings_dock, 
                              relativeTo=self.settings_dock, position='below')
        self.corr_dock = self.dockarea.addDock(name='correlation', 
                              position='right',  relativeTo = self.spec_dock)
        
        
        # Image View
        self.imview = pg.ImageView()
        self.imview.getView().invertY(False) # lower left origin
        self.image_dock.addWidget(self.imview)
        self.graph_layout = self.plot_n_fit.graph_layout

        # Rectangle ROI
        self.rect_roi = pg.RectROI([20, 20], [20, 20], pen=self.line_colors[0])
        self.rect_roi.addTranslateHandle((0.5,0.5))        
        self.imview.getView().addItem(self.rect_roi)        
        self.rect_roi.sigRegionChanged[object].connect(self.on_change_rect_roi)
        
        # Point ROI
        self.circ_roi = pg.CircleROI( (0,0), (2,2) , movable=True, pen=self.line_colors[1])
        #self.circ_roi.removeHandle(self.circ_roi.getHandles()[0])
        h = self.circ_roi.addTranslateHandle((0.5,.5))
        h.pen = pg.mkPen(pen=self.line_colors[1])
        h.update()
        self.imview.getView().addItem(self.circ_roi)
        self.circ_roi.removeHandle(0)
        self.circ_roi_plotline = pg.PlotCurveItem([0], pen=self.line_colors[1])
        self.imview.getView().addItem(self.circ_roi_plotline) 
        self.circ_roi.sigRegionChanged[object].connect(self.on_update_circ_roi)
        
        
        # Spec plot
        self.spec_plot = self.plot_n_fit.plot
        self.spec_plot.setLabel('left', 'Intensity', units='counts') 
        self.rect_plotdata = self.plot_n_fit.data_lines[0]
        self.point_plotdata = self.plot_n_fit.data_lines[1]
        self.point_plotdata.setZValue(-1)
                   
        
        #settings
        S = self.settings
        self.default_display_image_choices = ['default', 'sum']
        S.New('display_image', str, choices = self.default_display_image_choices, initial = 'default')    
        S.display_image.add_listener(self.on_change_display_image)    

        self.default_x_axis_choices = ['default', 'index']
        self.x_axis = S.New('x_axis', str, initial = 'default', choices = self.default_x_axis_choices)
        self.x_axis.add_listener(self.on_change_x_axis)      

        bg_subtract_choices = ('None', 'bg_slice', 'costum_const')
        self.bg_subtract = S.New('bg_subtract', str, initial='None', 
                                             choices=bg_subtract_choices)
        
        self.bg_counts = S.New('bg_value', initial=0, unit='cts/bin')
        self.bg_counts.add_listener(self.update_display)
        
        self.binning = S.New('binning', int, initial = 1, vmin=1)
        self.binning.add_listener(self.update_display)

        
        self.norm_data = S.New('norm_data', bool, initial = False)
        self.norm_data.add_listener(self.update_display)
        
        S.New('default_view_on_load', bool, initial=True)

        self.spatial_binning = S.New('spatial_binning', int, initial = 1, vmin=1)
        self.spatial_binning.add_listener(self.bin_spatially)

        self.show_lines = ['show_circ_line','show_rect_line']
        for x in self.show_lines:
            lq = S.New(x, bool, initial=True)
            lq.add_listener(self.on_change_show_lines)  


        # Settings Widgets
        self.settings_widgets = [] # Hack part 1/2: allows to use settings.New_UI() and have settings defined in scan_specific_setup()

        
        font = QtGui.QFont("Times", 12)
        font.setBold(True)
        self.x_slicer = RegionSlicer(self.spec_plot, name='x_slice', 
                                      #slicer_updated_func=self.update_display,
                                      brush = QtGui.QColor(0,255,0,50), 
                                      ZValue=10, font=font, initial=[100,511],
                                      activated=True)
        self.bg_slicer = RegionSlicer(self.spec_plot, name='bg_slice', 
                                      #slicer_updated_func=self.update_display,
                                      brush = QtGui.QColor(255,255,255,50), 
                                      ZValue=11, font=font, initial=[0,80], label_line=0)
        
                
        self.x_slicer.region_changed_signal.connect(self.update_display)
        self.bg_slicer.region_changed_signal.connect(self.update_display)
        
        self.bg_slicer.activated.add_listener(lambda:self.bg_subtract.update_value('bg_slice') if self.bg_slicer.activated.val else None)        
        self.settings_widgets.append(self.x_slicer.New_UI())
        self.settings_widgets.append(self.bg_slicer.New_UI())      

      
        ## Setting widgets, (w/o logged quantities)
        self.update_display_pushButton = QtWidgets.QPushButton(text = 'update display')
        self.settings_widgets.append(self.update_display_pushButton)
        self.update_display_pushButton.clicked.connect(self.update_display)  

        self.default_view_pushButton = QtWidgets.QPushButton(text = 'default img view')
        self.settings_widgets.append(self.default_view_pushButton)
        self.default_view_pushButton.clicked.connect(self.default_image_view) 
        
        self.recalc_median_pushButton = QtWidgets.QPushButton(text = 'recalc median map')
        self.settings_widgets.append(self.recalc_median_pushButton)
        self.recalc_median_pushButton.clicked.connect(self.recalc_median_map)

        self.recalc_sum_pushButton = QtWidgets.QPushButton(text = 'recalc sum map')
        self.settings_widgets.append(self.recalc_sum_pushButton)
        self.recalc_sum_pushButton.clicked.connect(self.recalc_sum_map)

        self.delete_current_display_image_pushButton = QtWidgets.QPushButton(text = 'delete image')
        self.settings_widgets.append(self.delete_current_display_image_pushButton)
        self.delete_current_display_image_pushButton.clicked.connect(self.delete_current_display_image)


        #correlation plot
        self.corr_layout = pg.GraphicsLayoutWidget()
        self.corr_plot = self.corr_layout.addPlot()
        self.corr_plotdata = pg.ScatterPlotItem(x=[0,1,2,3,4], y=[0,2,1,3,2], size=17, 
                                        pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 60))
        self.corr_plot.addItem(self.corr_plotdata)        
        self.corr_plotdata.sigClicked.connect(self.corr_plot_clicked)
        self.corr_dock.addWidget(self.corr_layout)

        self.corr_settings = CS = LQCollection()
        self.cor_X_data = self.corr_settings.New('cor_X_data', str, choices = self.default_display_image_choices,
                                            initial = 'default')
        self.cor_Y_data = self.corr_settings.New('cor_Y_data', str, choices = self.default_display_image_choices,
                                            initial = 'sum')
        self.cor_X_data.add_listener(self.on_change_corr_settings)
        self.cor_Y_data.add_listener(self.on_change_corr_settings)  
        self.corr_ui = self.corr_settings.New_UI()      
        self.corr_dock.addWidget(self.corr_ui)
        
        
        # map exporter       
        self.map_export_settings = MES = LQCollection()
        MES.New('include_scale_bar', bool, initial = True)
        MES.New('scale_bar_width', float, initial=1, spinbox_decimals = 3)
        MES.New('scale_bar_text', str, ro=False)
        map_export_ui = MES.New_UI()
        self.export_dock.addWidget( map_export_ui )
                
        self.export_maps_as_jpegs_pushButton = QtWidgets.QPushButton('export maps as jpegs')
        self.export_maps_as_jpegs_pushButton.clicked.connect(self.export_maps_as_jpegs)     
        self.export_dock.addWidget( self.export_maps_as_jpegs_pushButton )   
 
 
        self.save_state_pushButton = QtWidgets.QPushButton(text = 'save state')
        self.export_dock.addWidget(self.save_state_pushButton)
        self.save_state_pushButton.clicked.connect(self.save_state)              
      
                
        # finalize settings widgets
        self.scan_specific_setup() # there could more settings_widgets generated here (part 2/2)
                
                    
        hide_settings = ['norm_data', 'show_circ_line','show_rect_line',
                         'default_view_on_load', 'spatial_binning', 
                          'x_axis']        
        self.settings_ui = self.settings.New_UI(exclude=hide_settings)
        self.settings_dock.addWidget(self.settings_ui)   
        self.hidden_settings_ui =  self.settings.New_UI(include=hide_settings)
        self.export_dock.addWidget(self.hidden_settings_ui)
                     
        ui_widget =  QtWidgets.QWidget()
        gridLayout = QtWidgets.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        ui_widget.setLayout(gridLayout)        
        for i,w in enumerate(self.settings_widgets):
            gridLayout.addWidget(w, int(i/2), i%2)
        self.settings_dock.addWidget(ui_widget)          
        
                
        self.plot_n_fit.add_button('fit_map', self.fit_map)

        
        self.settings_dock.raiseDock()

        self.plot_n_fit.settings_dock.setStretch(1, 1)
        self.export_dock.setStretch(1,1)
        self.settings_dock.setStretch(1, 1)

        for layout in [self.settings_ui.layout(), self.export_dock.layout, ]:
            VSpacerItem = QtWidgets.QSpacerItem(0, 0,
                                                QtWidgets.QSizePolicy.Minimum,
                                                QtWidgets.QSizePolicy.Expanding)
            layout.addItem(VSpacerItem)

        
    def fit_map(self):
        x, hyperspec = self.get_xhyperspec_data(apply_use_x_slice=True)
        keys,images = self.plot_n_fit.fit_hyperspec(x, hyperspec)
        if len(keys) == 1:
            self.add_display_image(keys[0], images)
        else:
            for key, image in zip(keys, images):
                self.add_display_image(key, image)
               
                
    def add_spec_x_array(self, key, array):
        self.spec_x_arrays[key] = array
        self.settings.x_axis.add_choices(key, allow_duplicates=False)

    def add_display_image(self, key, image):
        key = self.add_descriptor_suffixes(key)
        self.display_images[key] = image
        self.settings.display_image.add_choices(key, allow_duplicates=False)
        self.cor_X_data.change_choice_list(self.display_images.keys())
        self.cor_Y_data.change_choice_list(self.display_images.keys())
        self.cor_X_data.update_value(self.cor_Y_data.val)
        self.cor_Y_data.update_value(key)
        self.on_change_corr_settings()
        print('added', key, image.shape)
    
    def add_descriptor_suffixes(self, key):
        if self.x_slicer.activated.val:
            key += '_x{}-{}'.format(self.x_slicer.start.val, self.x_slicer.stop.val)
        if self.settings['bg_subtract'] == 'bg_slice' and self.bg_slicer.activated.val:
            key += '_bg{}-{}'.format(self.bg_slicer.start.val, self.bg_slicer.stop.val)
        if self.settings['bg_subtract'] == 'costum_count':
            key += '_bg{1.2f}'.format(self.bg_counts.val)
        return key
    
    def delete_current_display_image(self):
        key = self.settings.display_image.val
        del self.display_images[key]
        self.settings.display_image.remove_choices(key)
        self.cor_X_data.remove_choices(key)
        self.cor_Y_data.remove_choices(key)

            
    def get_xy(self, ji_slice, apply_use_x_slice=False):
        '''
        returns processed hyperspec_data averaged over a given spatial slice.
        '''
        x,hyperspec_dat = self.get_xhyperspec_data(apply_use_x_slice)
        y = hyperspec_dat[ji_slice].mean(axis=(0,1))
        #self.databrowser.ui.statusbar.showMessage('get_xy(), counts in slice: {}'.format( y.sum() ) )

        if self.settings['norm_data']:
            y = norm(y)          
        return (x,y)
    
    def get_bg(self):
        bg_subtract_mode = self.bg_subtract.val
        if bg_subtract_mode == 'bg_slice' and hasattr(self, 'bg_slicer'):
            if not self.bg_slicer.activated:
                self.bg_slicer.activated.update_value(True)
            bg_slice = self.bg_slicer.slice
            bg = self.hyperspec_data[:,:,bg_slice].mean()
            self.bg_slicer.set_label(title=bg_subtract_mode,
                text='{:1.1f} cts<br>{} bins'.format(bg,bg_slice.stop-bg_slice.start))
        elif bg_subtract_mode == 'costum_const':
            bg = self.bg_counts.val
            self.bg_slicer.set_label('', title=bg_subtract_mode)
        else:
            bg = 0
            #self.bg_slicer.set_label('', title=bg_subtract_mode)
        return bg
        
    def get_xhyperspec_data(self, apply_use_x_slice=True):
        '''
        returns processed hyperspec_data
        '''
        bg = self.get_bg()
        hyperspec_data = self.hyperspec_data
        x = self.spec_x_array
        if apply_use_x_slice and self.x_slicer.activated.val:
            x = x[self.x_slicer.slice]
            hyperspec_data = hyperspec_data[:,:,self.x_slicer.slice]
        binning = self.settings['binning']
        if  binning!= 1:
            x,hyperspec_data = bin_y_average_x(x, hyperspec_data, binning, -1, datapoints_lost_warning=False)
            bg *= binning
        msg = 'effective subtracted bg value is binnging*bg ={:0.1f} which is up to {:2.1f}% of max value.'.format(bg, bg/np.max(hyperspec_data)*100 )
        self.databrowser.ui.statusbar.showMessage(msg)
        if self.settings['norm_data']:
            return (x,norm_map(hyperspec_data-bg))
        else:
            return (x,hyperspec_data-bg)
    
    def on_change_x_axis(self):
        key = self.settings['x_axis']
        if key in self.spec_x_arrays:
            self.spec_x_array = self.spec_x_arrays[key]
            self.x_slicer.set_x_array(self.spec_x_array)
            self.bg_slicer.set_x_array(self.spec_x_array)            
            self.spec_plot.setLabel('bottom', key)
            self.update_display()
            
    def on_change_display_image(self):
        key = self.settings['display_image']
        if key in self.display_images:
            self.display_image = self.display_images[key]
            self.update_display_image()
        if self.display_image.shape == (1,1):
            self.databrowser.ui.statusbar.showMessage('Can not display single pixel image!')
                
    def scan_specific_setup(self):
        #add settings and export_settings. Append widgets to self.settings_widgets and self.export_widgets
        pass
        
    def is_file_supported(self, fname):
        # override this!
        return False

    def post_load(self):
        # override this!
        pass    
    
    def on_change_data_filename(self, fname):
        self.reset()
        try:
            self.scalebar_type = None
            self.load_data(fname)
            if self.settings['spatial_binning'] != 1:
                self.hyperspec_data = bin_2D(self.hyperspec_data, self.settings['spatial_binning'])
                self.display_image = bin_2D(self.display_image, self.settings['spatial_binning'])
            print('on_change_data_filename', self.display_image.sum())
        except Exception as err:
            HyperSpectralBaseView.load_data(self, fname) # load default dummy data
            self.databrowser.ui.statusbar.showMessage("failed to load {}: {}".format(fname, err))
            raise(err)
        finally:
            self.display_images['default'] = self.display_image
            self.display_images['sum'] = self.hyperspec_data.sum(axis=-1)         
            self.spec_x_arrays['default'] = self.spec_x_array
            self.spec_x_arrays['index'] = np.arange(self.hyperspec_data.shape[-1])
            self.databrowser.ui.statusbar.clearMessage()
            self.post_load()
            self.add_scalebar()
            self.on_change_display_image()
            self.on_change_corr_settings()
            self.update_display()
        self.on_change_x_axis()


        print('loaded new file')
        if self.settings['default_view_on_load']:
            self.default_image_view()   
            
    def add_scalebar(self):
        ''' not intended to use: Call set_scalebar_params() during load_data()'''
        
        if hasattr(self, 'scalebar'):
            self.imview.getView().removeItem(self.scalebar)
            del self.scalebar
                
        num_px = self.display_image.shape[1] #horizontal dimension!

        if self.scalebar_type == None: 
            #matplotlib export
            self.unit_per_px = 1
            self.map_export_settings['scale_bar_width'] = int(num_px/4)
            self.map_export_settings['scale_bar_text'] = '{} pixels'.format(int(num_px/4))          

            
        if self.scalebar_type != None: 
            kwargs = self.scalebar_kwargs    
            
            span = self.scalebar_kwargs['span'] # this is in meter! convert to according to its magnitude
            w_meter = span / 4
            mag = int(np.log10(w_meter))
            conv_fac, unit = {0: (1,'m'), 
                        -1:(1e2,'cm'),-2:(1e3,'mm'), -3:(1e3,'mm'),
                        -4:(1e6,'\u03bcm'),-5:(1e6,'\u03bcm'), -6:(1e6,'\u03bcm'), #\mu
                        -7:(1e9,'nm'),-8:(1e9,'nm'), -9:(1e9,'nm'),
                        -10:(1e10,'\u212b'),
                        -11:(1e12,'pm'), -12:(1e12,'pm')}[mag]
                                     
            #matplotlib export           
            self.unit_per_px = span * conv_fac / num_px
            self.map_export_settings['scale_bar_width'] = int(w_meter * conv_fac)
            self.map_export_settings['scale_bar_text'] = f'{int(w_meter * conv_fac)} {(unit)}'
        
            
        if self.scalebar_type == 'ConfocalScaleBar':
            self.scalebar = ConfocalScaleBar(num_px=num_px, 
                                **kwargs)
            self.scalebar.setParentItem(self.imview.getView())
            self.scalebar.anchor((1, 1), (1, 1), offset=kwargs['offset'])


        elif self.scalebar_type == None:
            self.scalebar = None

            
            
    def set_scalebar_params(self, h_span, units='m', scalebar_type='ConfocalScaleBar',
                           stroke_width=10, brush='w', pen='k', offset=(-20, -20)):
        '''
        call this function during load_data() to add a scalebar!
        *h_span*  horizontal length of image in units of *units* if positive.
                  Else, scalebar is in units of pixels (*units* ignored).
        *units*   SI length unit of *h_span*.
        *scalebar_type* is either `None` (no scalebar will be added)
          or `"ConfocalScaleBar"` (default).
        *stroke_width*, *brush*, *pen* and *offset* affect appearance and 
         positioning of the scalebar.
        '''
        assert scalebar_type in [None, 'ConfocalScaleBar']
        self.scalebar_type = scalebar_type
        span_meter = {'m':1, 'cm':1e-2, 'mm':1e-3, 'um':1e-6, 
                      'nm':1e-9, 'pm':1e-12, 'fm':1e-15}[units] * h_span
        self.scalebar_kwargs = {'span':span_meter, 'brush':brush, 'pen':pen,
                                'width':stroke_width, 'offset':offset}
        


    @QtCore.Slot()
    def update_display(self):
        # pyqtgraph axes are (x,y), but display_images are in (y,x) so we need to transpose        
        if self.display_image is not None:
            self.update_display_image()
            self.on_change_rect_roi()
            self.on_update_circ_roi()
            
    def update_display_image(self):
        if self.display_image is not None:
            self.imview.setImage(self.display_image.T)  
                      
    def reset(self):
        '''
        resets the dictionaries
        '''
        keys_to_delete = list( set(self.display_images.keys()) - set(self.default_display_image_choices) )
        for key in keys_to_delete:
            del self.display_images[key]
        keys_to_delete = list( set(self.spec_x_arrays.keys()) - set(self.default_x_axis_choices) )
        for key in keys_to_delete:
            del self.spec_x_arrays[key]
        self.settings.display_image.change_choice_list(self.default_display_image_choices)
        self.settings.x_axis.change_choice_list(self.default_x_axis_choices)

    
    def load_data(self, fname):
        """
        override to set hyperspectral dataset and the display image
        need to define:
            * self.hyperspec_data (shape Ny, Nx, Nspec)
            * self.display_image (shape Ny, Nx)
            * self.spec_x_array (shape Nspec)
        """
        self.hyperspec_data = np.arange(10*10*34).reshape( (10,10,34) )
        self.display_image = self.hyperspec_data.sum(-1)
        self.spec_x_array = np.arange(34)
    
    @QtCore.Slot(object)
    def on_change_rect_roi(self, roi=None):
        # pyqtgraph axes are (x,y), but hyperspec is in (y,x,spec) hence axes=(1,0)      
        roi_slice, roi_tr = self.rect_roi.getArraySlice(self.hyperspec_data, self.imview.getImageItem(), axes=(1,0)) 
        self.rect_roi_slice = roi_slice
        
        x,y = self.get_xy(self.rect_roi_slice, apply_use_x_slice=False)  
        self.plot_n_fit.update_data(x, y, 0, is_fit_data=False)

        x_fit_data, y_fit_data = self.get_xy(self.rect_roi_slice, apply_use_x_slice=True)  
        self.plot_n_fit.update_fit_data(x_fit_data, y_fit_data)
        text = self.plot_n_fit.result_message
        title = self.plot_n_fit.state_info + ' rect'
        self.x_slicer.set_label(text, title, color = self.line_colors[0])

        self.on_change_corr_settings()

        
    @QtCore.Slot(object)        
    def on_update_circ_roi(self, roi=None):
        if roi is None:
            roi = self.circ_roi

        roi_state = roi.saveState()
        x0, y0 = roi_state['pos']
        xc = x0 + 1
        yc = y0 + 1

        Ny, Nx, Nspec = self.hyperspec_data.shape
        
        i = max(0, min(int(xc),  Nx-1))
        j = max(0, min(int(yc),  Ny-1))
                
        self.circ_roi_plotline.setData([xc, i+0.5], [yc, j + 0.5])
        
        self.circ_roi_ji = (j,i)    
        self.circ_roi_slice = np.s_[j:j+1,i:i+1]
        
        x,y = self.get_xy(self.circ_roi_slice, apply_use_x_slice=False)
        self.plot_n_fit.update_data(x, y, 1, is_fit_data=False)
        
        x_fit_data, y_fit_data = self.get_xy(self.circ_roi_slice, apply_use_x_slice=True)  
        self.plot_n_fit.update_fit_data(x_fit_data, y_fit_data)
        text = self.plot_n_fit.result_message
        title = self.plot_n_fit.state_info + ' circ'
        self.x_slicer.set_label(text, title, color = self.line_colors[1])
        
        self.on_change_corr_settings()


    def on_change_show_lines(self):
        self.point_plotdata.setVisible(self.settings['show_circ_line'])
        self.rect_plotdata.setVisible(self.settings['show_rect_line'])
 
        
    def default_image_view(self):
        'sets rect_roi congruent to imageItem and optimizes size of imageItem to fit the ViewBox'
        iI = self.imview.imageItem
        h,w  = iI.height(), iI.width()       
        self.rect_roi.setSize((w,h))
        self.rect_roi.setPos((0,0))
        self.imview.getView().enableAutoRange()
        self.spec_plot.enableAutoRange()
        
    def recalc_median_map(self):
        x,hyperspec_data = self.get_xhyperspec_data(apply_use_x_slice=True)
        median_map = spectral_median_map(hyperspec_data,x)
        self.add_display_image('median_map', median_map)
        
    def recalc_sum_map(self):
        _,hyperspec_data = self.get_xhyperspec_data(apply_use_x_slice=True)
        _sum = hyperspec_data.sum(-1)
        self.add_display_image('sum', _sum)
        
          
    def on_change_corr_settings(self):
        try:
            xname = self.corr_settings['cor_X_data']
            yname = self.corr_settings['cor_Y_data']
            X = self.display_images[xname]
            Y = self.display_images[yname]

            #Note, the correlation plot is a dimensionality reduction 
            # (i,j,X,Y) --> (X,Y). To map the scatter points back to the image
            # we need to associate every (X,Y) on the correlation plot with 
            # their indices (i,j); in particular 
            # indices = [(j0,i0), (j0,i1), ...]
            indices = list( np.indices((X.shape)).reshape(2,-1).T )
            self.corr_plotdata.setData(X.flat, Y.flat, brush=pg.mkBrush(255, 255, 255, 50),
                                       pen=None, data=indices)

            # mark points within rect_roi 
            mask = np.zeros_like(X, dtype=bool)
            mask[self.rect_roi_slice[0:2]] = True
            cor_x = X[mask].flatten()
            cor_y = Y[mask].flatten()
            self.corr_plotdata.addPoints(cor_x, cor_y, brush=pg.mkBrush(255, 255, 204, 60), 
                    pen=pg.mkPen(self.line_colors[0], width=0.5))
            
            # mark circ_roi point
            j,i = self.circ_roi_ji
            x_circ, y_circ = np.atleast_1d(X[j,i]), np.atleast_1d(Y[j,i])
            self.corr_plotdata.addPoints(x=x_circ, y=y_circ, 
                    pen=pg.mkPen(self.line_colors[1], width=3))

            ##some more plot details 
            #self.corr_plot.getViewBox().setRange(xRange=(cor_x.min(), cor_x.max()),
            #                                     yRange=(cor_y.min(), cor_y.max()))
    
            self.corr_plot.autoRange()
            self.corr_plot.setLabels(**{'bottom':xname,'left':yname})
            sm = spearmanr(cor_x, cor_y)
            text = 'Pearson\'s corr: {:.3f}<br>Spearman\'s: corr={:.3f}, pvalue={:.3f}'.format(
                                np.corrcoef(cor_x,cor_y)[0,1], sm.correlation, sm.pvalue)
            self.corr_plot.setTitle(text)
            
        except Exception as err:
            print('Error in on_change_corr_settings: {}'.format(err))
            self.databrowser.ui.statusbar.showMessage('Error in on_change_corr_settings: {}'.format(err))

    def corr_plot_clicked(self, plotitem, points):
        '''
        call back function to locate a point on the correlation plot on the image. 
        
        *points* is a list of <pg.ScatterPlotItem.SpotItem> under the mouse 
        pointer during click event. For points within the rect_roi, there are 
        two items representing a pixel, but only most button one contains the
        (j,i) as data.    
        '''
        j,i = points[-1].data()         
        self.circ_roi.setPos(i-0.5,j-0.5)
                
        
    def bin_spatially(self):
        if not (self.settings['display_image'] in self.default_display_image_choices):
            self.settings.display_image.update_value( self.default_display_image_choices[0] )
        fname = self.databrowser.settings['data_filename']
        self.on_change_data_filename(fname)
    
    def export_maps_as_jpegs(self):
        for name,image in self.display_images.items():
            self.export_image_as_jpeg(name, image)   
    
    def export_image_as_jpeg(self, name, image, cmap='gist_heat'):
        import matplotlib.pylab as plt
        plt.figure(dpi=200)
        plt.title(name)        
        ax = plt.subplot(111)
        Ny, Nx = image.shape
        extent = [0, self.unit_per_px * Nx, 0, self.unit_per_px * Ny]   
        plt.imshow(image, origin='lower', interpolation=None, cmap=cmap, 
                   extent=extent,
                   )
        
        ES = self.map_export_settings
        if ES['include_scale_bar']:
            add_scale_bar(ax, ES['scale_bar_width'], ES['scale_bar_text'])
        cb = plt.colorbar()
        plt.tight_layout()
        fig_name =  self.fname.replace('.h5', '_{:0.0f}_{}.jpg'.format(time.time(), name)) 
        plt.savefig(fig_name)
        plt.close()
    
    def save_state(self):
        from ScopeFoundry import h5_io
        fname = self.databrowser.settings['data_filename']
        view_state_fname = '{fname}_state_view_{timestamp:%y%m%d_%H%M%S}.{ext}'.format(
            fname = fname.strip('.h5'),
            timestamp=datetime.fromtimestamp(time.time()),
            ext='h5')
        h5_file = h5py.File(name = view_state_fname)
        
        with h5_file as h5_file:
            h5_group_display_images = h5_file.create_group('display_images')
            for k,v in self.display_images.items():
                h5_group_display_images.create_dataset(k, data=v)
            h5_group_spec_x_array = h5_file.create_group('spec_x_arrays')
            for k,v in self.spec_x_arrays.items():
                h5_group_spec_x_array.create_dataset(k, data=v)
            h5_group_settings_group = h5_file.create_group('settings')
            h5_io.h5_save_lqcoll_to_attrs(self.settings, h5_group_settings_group)
            h5_group_settings_group = h5_file.create_group('x_slicer_settings')
            h5_io.h5_save_lqcoll_to_attrs(self.x_slicer.settings, h5_group_settings_group)            
            h5_group_settings_group = h5_file.create_group('bg_slicer_settings')
            h5_io.h5_save_lqcoll_to_attrs(self.bg_slicer.settings, h5_group_settings_group)
            h5_group_settings_group = h5_file.create_group('export_settings')
            h5_io.h5_save_lqcoll_to_attrs(self.export_settings, h5_group_settings_group)
            self.view_specific_save_state_func(h5_file)
            h5_file.close()

    def view_specific_save_state_func(self, h5_file):
        '''
        you can override me, use 'h5_file' - it's already open 
        e.g:  h5_file.create_group('scan_specific_settings')
         ...
        '''
        pass
    
    def load_state(self, fname_idx=-1):
        
        # does not work properly, maybe because the order the settings are set matters?
        path = sibling_path(self.databrowser.settings['data_filename'], '')
        pre_state_fname = self.databrowser.settings['data_filename'].strip(path).strip('.h5')
        
        state_files = []
        for x in os.listdir(path):
            if pre_state_fname in x:
                if 'state_view' in x:
                    state_files.append(x)
            
        print('state_files', state_files)
        
        if len(state_files) != 0:
            h5_file = h5py.File(path + state_files[fname_idx])
            for k,v in h5_file['bg_slicer_settings'].attrs.items():
                try:
                    self.bg_slicer.settings[k] = v
                except:
                    pass    
            for k,v in h5_file['x_slicer_settings'].attrs.items():
                try:
                    self.x_slicer.settings[k] = v
                except:
                    pass
                
            for k,v in h5_file['settings'].attrs.items():
                try:
                    self.settings[k] = v
                except:
                    pass

            for k,v in h5_file['biexponential_settings'].attrs.items():
                self.biexponential_settings[k] = v
            for k,v in h5_file['export_settings'].attrs.items():
                self.export_settings[k] = v
            
            h5_file.close()
            print('loaded', state_files[fname_idx])
            
            
            
            
            
            
            
def spectral_median(spec, wls, count_min=200):
    int_spec = np.cumsum(spec)
    total_sum = int_spec[-1]
    if total_sum > count_min:
        pos = int_spec.searchsorted( 0.5*total_sum)
        wl = wls[pos]
    else:
        wl = 0
    return wl
def spectral_median_map(hyperspectral_data, wls):
    return np.apply_along_axis(spectral_median, -1, hyperspectral_data, wls=wls)  
def norm(x):
    x_max = x.max()
    if x_max==0:
        return x*0.0
    else:
        return x*1.0/x_max
def norm_map(map_):
    return np.apply_along_axis(norm, -1, map_)
    
    
def bin_y_average_x(x, y, binning = 2, axis = -1, datapoints_lost_warning = True):
    '''
    y can be a n-dim array with length on axis `axis` equal to len(x)
    '''    
    new_len = int(x.__len__()/binning) * binning
    
    data_loss = x.__len__() - new_len
    if data_loss is not 0 and datapoints_lost_warning:
        print('bin_y_average_x() warining: lost final', data_loss, 'datapoints')
    
    def bin_1Darray(arr, binning=binning, new_len=new_len):
        return arr[:new_len].reshape((-1,binning)).sum(1)
    
    x_ = bin_1Darray(x) / binning
    y_ = np.apply_along_axis(bin_1Darray,axis,y)
    
    return x_, y_

def bin_2D(arr,binning=2):
    '''
    bins an array of at least 2 dimension along the axis 0 and 1
    '''
    shape = arr.shape
    new_dim = int(shape[0]/binning)
    salvaged_along_dim = new_dim*binning
    lost_lines_0 = shape[0]-salvaged_along_dim
    arr = arr[0:salvaged_along_dim].reshape((-1,binning,shape[1],*shape[2:])).sum(1)
    shape = arr.shape
    new_dim = int(shape[1]/binning)
    salvaged_along_dim = new_dim*binning
    lost_lines_1 = shape[1]-salvaged_along_dim
    arr = arr[:,0:salvaged_along_dim].reshape((shape[0],-1,binning,*shape[2:])).sum(2)
    if (lost_lines_1 + lost_lines_0) >0 :
        print('cropped data:', (lost_lines_0,lost_lines_1), 'lines lost' )
    return arr



def add_scale_bar(ax, width=0.005, text=True, d=None, height=None, h_pos='left', v_pos='bottom',
                  color='w', edgecolor='k', lw=1, set_ticks_off=True, origin_lower=True, fontsize=13):
    from matplotlib.patches import Rectangle
    import matplotlib.pylab as plt
    imshow_ticks_off_kwargs = dict(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                         labelbottom=False, labeltop=False, labelleft=False, labelright=False)  
    """
        places a rectancle onto the axis *ax.
        d is the distance from the edge to rectangle.
    """
    
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    x1, y1 = ax.get_xlim()[1], ax.get_ylim()[1]
    
    Dx = x1 - x0
    if d == None:
        d = Dx / 18.
    if height == None:
        height = d * 0.8
    if width == None:
        width = 5 * d

    if h_pos == 'left':
        X = x0 + d
    else:
        X = x1 - d - width

    
    if origin_lower:
        if v_pos == 'bottom':
            Y = y0 + d
        else:
            Y = y1 - d - height
    else:
        if v_pos == 'bottom':
            Y = y0 - d - height
        else:
            Y = y1 + d

    xy = (X, Y)
    
    p = Rectangle(xy, width, height, color=color, ls='solid', lw=lw, ec=edgecolor)
    ax.add_patch(p)
    
    
    if text:
        if type(text) in [bool, None] or text == 'auto':
            text = str(int(width*1000)) + ' \u03BCm'
            print('caution: Assumes extent to be in mm, set text arg manually!')
        if v_pos == 'bottom':
            Y_text = Y+1.1*d
            va = 'bottom'
        else:
            Y_text = Y-0.1*d
            va = 'top'
        txt = plt.text(X+0.5*width,Y_text,text,
                 fontdict={'color':color, 'weight': 'heavy', 'size':fontsize,
                           #'backgroundcolor':edgecolor, 
                           'alpha':1, 'horizontalalignment':'center', 'verticalalignment':va}
                )
        import matplotlib.patheffects as PathEffects
        txt.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground=edgecolor)])    
    
    if set_ticks_off:
        ax.tick_params(**imshow_ticks_off_kwargs)