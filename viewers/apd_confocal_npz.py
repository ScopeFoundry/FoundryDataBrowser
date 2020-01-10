from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
import pyqtgraph as pg
import numpy as np
from qtpy import QtWidgets

class ApdConfocalNPZView(DataBrowserView):

    name = 'apd_confocal_npz'
    
    def setup(self):
        
        self.ui = self.imview = pg.ImageView()
        
        #self.graph_layout = pg.GraphicsLayoutWidget()
        #self.graph_layout.addPlot()
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = np.load(fname)
            self.imview.setImage(self.dat['count_rate_map'][::-1,:].T)
        except Exception as err:
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return "apd_confocal.npz" in fname

class ApdConfocal3dNPZView(DataBrowserView):

    name = 'apd_confocal_3d_npz'
    
    def setup(self):
        
        self.settings.New('plane', dtype=str, initial='xy', choices=('xy', 'yz', 'xz'))
        self.settings.New('index', dtype=int)
        self.settings.New('auto_level', dtype=bool, initial=True)
        for name in ['plane', 'index', 'auto_level']:
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
            self.dat = np.load(fname)
            self.update_display()
        except Exception as err:
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return "apd_confocal_scan3d.npz" in fname
    
    def update_display(self):
        
        if hasattr(self,"dat"):
        
            ii = self.settings['index']
            plane = self.settings['plane']
            
            if plane == 'xy':        
                arr_slice = np.s_[ii,:,:]
                index_max = self.dat['count_rate_map'].shape[0]
            elif plane == 'yz':
                arr_slice = np.s_[:,:,ii]
                index_max = self.dat['count_rate_map'].shape[2]
            elif plane == 'xz':
                arr_slice = np.s_[:,ii,:]
                index_max = self.dat['count_rate_map'].shape[1] 
    
            self.settings.index.change_min_max(0, index_max)
            
            
            self.imview.setImage(self.dat['count_rate_map'][arr_slice], autoLevels=self.settings['auto_level'], )
    
            other_ax = dict(xy='z', yz='x', xz='y' )[plane]
    
            self.info_label.setText("{} plane {}={} um (index={})".format(
                plane, other_ax, self.dat[other_ax+'_array'][ii], ii))


        
if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(ApdConfocalNPZView(app))
    
    sys.exit(app.exec_())