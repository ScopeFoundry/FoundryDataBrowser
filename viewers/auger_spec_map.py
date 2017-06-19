from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg
import pyqtgraph.dockarea as dockarea
from qtpy import QtWidgets, QtGui


class AugerSpecMapView(DataBrowserView):
    
    name = 'auger_spec_map'
    
    def setup(self):
        
        self.settings.New('ke0_start', dtype=float)
        self.settings.New('ke0_stop', dtype=float)
        self.settings.New('ke1_start', dtype=float)
        self.settings.New('ke1_stop', dtype=float)
        
        self.settings.New('math_mode', dtype=str, initial='A',
                          choices=('A','B','A+B', 'A-B', 'B-A'))
        
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
            
            norm_chans = np.array([ 1.        ,  1.04156986,  1.14521567,  0.9713133 ,  1.13392579,
        0.9678894 ,  1.18444014])
            norm_chans = norm_chans.reshape(1,1,1,1,-1)
            self.auger_map *= norm_chans
            
            
            self.auger_sum_map = self.auger_map[:,:,:,:,0:7].mean(axis=4)
            self.ke = np.array(self.H['ke']) 
            
            scan_shape = self.adc_map.shape[:-1]
            
            # Display
            self.total_plotline.setData(*self.compute_total_spectrum())
            for ii in range(7):
                self.chan_plotlines[ii].setData(self.ke[ii,:],
                                                self.auger_map[:,:,:,:,ii].mean(axis=(1,2,3)))

            self.on_change_ke_settings()
            
            #self.update_display()
        except Exception as err:
            print(err)
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("Failed to load %s: %s" %(fname, err))
            raise(err)
    
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
        A = auger_ke0_imgs.mean(axis=0)
        B = auger_ke1_imgs.mean(axis=0)
        
        self.imview_auger.setImage(self.compute_image(A,B))
        
        self.lr0.setRegion((S['ke0_start'], S['ke0_stop']))
        self.lr1.setRegion((S['ke1_start'], S['ke1_stop']))
            
    def on_change_regions(self):
        S = self.settings
        S['ke0_start'], S['ke0_stop'] = self.lr0.getRegion()
        S['ke1_start'], S['ke1_stop'] = self.lr1.getRegion()
    
    def compute_image(self, A,B):
        mm = self.settings['math_mode']
        if mm == 'A': return A
        if mm == 'B': return B
        if mm == 'A+B': return A+B
        if mm == 'A-B': return A-B
        if mm == 'B-A': return B-A
        
        
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

    def compute_total_spectrum(self):
        from scipy import interpolate
        total_spec = np.zeros(100, dtype=float)
        ke_interp = np.linspace(self.ke.min(), self.ke.max(), 100, dtype=float)
        for i in range(0,7):
            x = self.ke[i,:]
            y = self.auger_map[:,:,:,:,i].mean(axis=(1,2,3))
            ff = interpolate.interp1d(x,y,bounds_error=False)
            total_spec += ff(ke_interp)
        return ke_interp, (total_spec/7.0)

