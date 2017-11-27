from ScopeFoundry.data_browser import DataBrowserView
import pyqtgraph as pg
import numpy as np
from qtpy import QtWidgets
import h5py

class AugerSyncRasterScanH5(DataBrowserView):

    name = 'auger_sync_raster_scan'
    
    def setup(self):
        
        self.settings.New('frame', dtype=int)
        #self.settings.New('sub_frame', dtype=int)
        self.settings.New('source', dtype=str, choices=('SEM', 'Auger'))
        self.settings.New('SEM_chan', dtype=int, vmin=0, vmax=1)
        self.settings.New('Auger_chan', dtype=int, vmin=0, vmax=7)
        self.settings.New('auto_level', dtype=bool, initial=True)
        for name in ['frame', 'source','SEM_chan', 'Auger_chan', 'auto_level']:
            self.settings.get_lq(name).add_listener(self.update_display)
        
        self.ui = QtWidgets.QWidget()
        self.ui.setLayout(QtWidgets.QVBoxLayout())
        self.ui.layout().addWidget(self.settings.New_UI(), stretch=0)
        self.info_label = QtWidgets.QLabel()
        self.ui.layout().addWidget(self.info_label, stretch=0)
        self.imview = pg.ImageView()
        self.ui.layout().addWidget(self.imview, stretch=1)
        
        #self.graph_layout = pg.GraphicsLayoutWidget()
        #self.graph_layout.addPlot()
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = h5py.File(fname)
            M = self.measurement = self.dat['measurement/auger_sync_raster_scan']
            nframe, nsubframe, ny, nx, nadc_chan = M['adc_map'].shape
            self.settings.frame.change_min_max(0, nframe-1)
            self.settings.sub_frame.change_min_max(0, nsubframe-1)
            self.update_display()
        except Exception as err:
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return "auger_sync_raster_scan.h5" in fname
    
    def update_display(self):
        
        M = self.measurement
        
        ke = M['ke']
        ii = self.settings['frame']
        jj = 0 #self.settings['sub_frame']
        source = self.settings['source']
        if source=='SEM':
            kk = self.settings['SEM_chan']            
            im = M['adc_map'][ii, jj, :,:, kk]
            ke_info = " ke {:.1f} eV".format(ke[0,ii])
        else:
            kk = self.settings['Auger_chan']            
            im = M['auger_chan_map'][ii, jj, :,:, kk]
            ke_info = " ke {:.1f} eV".format(ke[kk,ii])

        self.imview.setImage(im.T, autoLevels=self.settings['auto_level'], )
        info = "Frame {} {} chan {} ".format(ii,source, kk)
        
        self.info_label.setText(info+ke_info)
        