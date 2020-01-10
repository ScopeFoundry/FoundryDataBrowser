from ScopeFoundry.data_browser import DataBrowser, HyperSpectralBaseView
import numpy as np
from qtpy import QtWidgets
import pyqtgraph as pg

class TRPLNPZView(HyperSpectralBaseView):

    name = 'trpl_scan_npz'
    
    def is_file_supported(self, fname):
        return "_trpl_scan.npz" in fname

    def load_data(self, fname):
        self.dat = np.load(fname)
        
        try:
            cr0 = self.dat['picoharp_count_rate0']
            rep_period_s = 1.0/cr0
            time_bin_resolution = self.dat['picoharp_Resolution']*1e-12
            self.num_hist_chans = int(np.ceil(rep_period_s/time_bin_resolution))
        except:
            self.num_hist_chans = self.time_trace_map.shape[-1]
        # truncate data to only show the time period associated with rep-rate of laser
        
        self.time_trace_map = self.dat['time_trace_map']
        self.integrated_count_map = self.dat['integrated_count_map']
        self.time_array = self.dat['time_array']

        self.hyperspec_data = self.time_trace_map[:,:,0:self.num_hist_chans]+1
        self.display_image = self.integrated_count_map
        self.spec_x_array = self.time_array[0:self.num_hist_chans]
    
    def scan_specific_setup(self):
        # set spectral plot to be semilog-y
        self.spec_plot.setLogMode(False, True)
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'time', units='ns')
    

class TRPL3dNPZView(HyperSpectralBaseView):

    name = 'trpl_3d_npz'
    
    def setup(self):
        HyperSpectralBaseView.setup(self)
        TRPLNPZView.scan_specific_setup(self)
        
        self.settings.New('plane', dtype=str, initial='xy', choices=('xy', 'yz', 'xz'))
        self.settings.New('index', dtype=int)
        self.settings.New('auto_level', dtype=bool, initial=True)
        
        #self.ui = QtWidgets.QWidget()
        #self.ui.setLayout(QtWidgets.QVBoxLayout())
        self.dockarea.addDock(name='Image', widget=self.settings.New_UI())
        self.info_label = QtWidgets.QLabel()
        self.dockarea.addDock(name='info', widget=self.info_label)
        #self.imview = pg.ImageView()
        #self.ui.layout().addWidget(self.imview, stretch=1)
        
        #self.graph_layout = pg.GraphicsLayoutWidget()
        #self.graph_layout.addPlot()

        for name in ['plane', 'index', 'auto_level']:
            self.settings.get_lq(name).add_listener(self.update_display)

    
    def load_data(self, fname):
        TRPLNPZView.load_data(self, fname)

#     def on_change_data_filename(self, fname):
#         
#         try:
#             TRPLNPZView.load_data(self, fname)
#             self.update_display()
#         except Exception as err:
#             self.imview.setImage(np.zeros((10,10)))
#             self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
#             raise(err)
        
    def is_file_supported(self, fname):
        return "trpl_scan3d.npz" in fname
    
    def update_display(self):
        if not hasattr(self, 'dat'):
            return 
        
        ii = self.settings['index']
        plane = self.settings['plane']
        
        if plane == 'xy':        
            arr_slice = np.s_[ii,:,:]
            index_max = self.dat['integrated_count_map'].shape[0]-1
        elif plane == 'yz':
            arr_slice = np.s_[:,:,ii]
            index_max = self.dat['integrated_count_map'].shape[2]-1
        elif plane == 'xz':
            arr_slice = np.s_[:,ii,:]
            index_max = self.dat['integrated_count_map'].shape[1]-1 

        self.settings.index.change_min_max(0, index_max)
        
        self.hyperspec_data = self.time_trace_map[:,:,:,0:self.num_hist_chans][arr_slice]+1
        self.display_image = self.integrated_count_map[arr_slice]
        
        
        #self.imview.setImage(self.dat['integrated_count_map'][arr_slice], autoLevels=self.settings['auto_level'], )

        other_ax = dict(xy='z', yz='x', xz='y' )[plane]

        self.info_label.setText("{} plane {}={} um (index={})".format(
            plane, other_ax, self.dat[other_ax+'_array'][ii], ii))
        
        HyperSpectralBaseView.update_display(self)

        

if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(TRPLNPZView(app))
    
    sys.exit(app.exec_())