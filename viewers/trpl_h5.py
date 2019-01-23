from ScopeFoundry.data_browser import DataBrowser, HyperSpectralBaseView
import numpy as np
import h5py
from qtpy import QtWidgets
import pyqtgraph as pg



class TRPLH5View(HyperSpectralBaseView):

    name = 'trpl_scan_h5'
    
        
    def is_file_supported(self, fname):
        for name in ["_trpl_2d_scan.h5", "_trpl_scan.h5"]:
            if name in fname: return True
        return False

    def load_data(self, fname):
        self.file = h5py.File(fname)
        
        load_success = False
        for measure_path in ['measurement/trpl_scan/', 'measurement/trpl_2d_scan/']:
            if  measure_path in self.file:
                self.H = self.file[measure_path]
                load_success = True
        if not load_success:
            print(self.H.items())
            raise ValueError("Measurement group not found in h5 file")
        
        self.S = self.file['hardware/picoharp/settings'].attrs
        
        try:
            cr0 = self.S['count_rate0']
            rep_period_s = 1.0/cr0
            time_bin_resolution = self.S['Resolution']*1e-12
            self.num_hist_chans = int(np.ceil(rep_period_s/time_bin_resolution))
        except:
            self.num_hist_chans = self.time_trace_map.shape[-1]
        
        
        t_slice = np.s_[0:self.num_hist_chans]
        
        time_array = np.array(self.H['time_array'])[t_slice]
        time_trace_map = np.array(self.H['time_trace_map'])[0,:,:,t_slice]
        integrated_count_map = time_trace_map.sum(axis=-1)

        # set defaults
        self.hyperspec_data = time_trace_map
        self.display_image = integrated_count_map
        self.spec_x_array = time_array

        self.file.close()
        
    def post_load(self):
        self.recalc_taue_map() 
                
    def scan_specific_setup(self):
        # set spectral plot to be semilog-y
        self.spec_plot.setLogMode(False, True)
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'time', units='ns')
        
        self.fit_on = self.settings.New('fit_on', str, initial = 'circ',
                                          choices = ('None', 'rect_roi', 'circ'))
        self.fit_on.add_listener(self.on_change_fit_on)
        self.fit_line = self.spec_plot.plot(pen='g')

        self.fit_option = self.settings.New('fit_option',str,initial='poly_fit',
                                             choices = ('poly_fit','tau_x_calc'))
        self.fit_option.add_listener(self.on_change_fit_option)
        
        self.fit_map_pushButton = QtWidgets.QPushButton(text = 'fit_map')
        self.settings_widgets.append(self.fit_map_pushButton)
        self.fit_map_pushButton.clicked.connect(lambda x:self.fit_map(update_display_image=True, fit_option=None))  
        
        
    def recalc_taue_map(self):
        self.fit_map(fit_option='tau_x_calc')
        
    def on_change_fit_option(self):
        self.fit_xy()
        
    def on_change_fit_on(self):
        if self.fit_on.val == 'None':
            alpha = 0
            self.x_slice_InfLineLabel.setText('x_slice')
        else:
            alpha = 1
        self.fit_line.setOpacity(alpha)
        self.fit_xy()

    def on_change_rect_roi(self):
        HyperSpectralBaseView.on_change_rect_roi(self)        
        self.fit_xy()

    def on_update_circ_roi(self):
        HyperSpectralBaseView.on_update_circ_roi(self)
        self.fit_xy()
        
    def fit_xy(self):
        if self.fit_on.val == 'rect_roi':
            x,y = self.get_xy(self.rect_roi_slice, apply_use_x_slice=True)
        elif self.fit_on.val == 'circ':
            x,y = self.get_xy(self.circ_roi_slice, apply_use_x_slice=True)
        else:
            return
        fit_func_dict = {'poly_fit':  self.poly_fit_xy,
                         'tau_x_calc': self.tau_x_calc_xy}
        fit_option = self.settings['fit_option']
        fit_func_dict[fit_option](x,y)
        
    def fit_map(self, update_display_image = True, fit_option = None):
        time_array, time_trace_map = self.get_xhyperspec_data(apply_use_x_slice=True)
        fit_func_dict = {'poly_fit':  self.poly_fit_map,
                         'tau_x_calc': self.tau_x_calc_map}
        if fit_option == None:
            fit_option = self.settings['fit_option']
        fit_map = fit_func_dict[fit_option](time_array, time_trace_map)
        new_map_name = fit_option+'_map'
        self.add_display_image(new_map_name, fit_map)
        if update_display_image:
            self.settings['display_image'] = new_map_name

    def tau_x_calc_xy(self,x,y):
        t = x.copy()
        t -= t.min()        
        tau = tau_x_calc(y, t)
        label_text = '\n'.join(['tau_x_calc', 'tau={:3.3f}'.format(tau)])
        self.x_slice_InfLineLabel.setText(label_text)
        self.fit_line.setData(x,y)
        return x,y
        #print('tau_x_calc', tau)

    def tau_x_calc_map(self, time_array, time_trace_map):
        return tau_x_calc_map(time_array - time_array.min(), time_trace_map)
        
    def poly_fit_xy(self,x,y,deg=1):       
        coefs = poly_fit(x=x, y=y)
        t = x - x.min()
        fit = np.exp( np.poly1d(coefs)(t) )    
        label_text = '\n'.join(['poly_fit','A={:3.3f}'.format(coefs[1]), 
                                'tau={:3.3f}'.format(-1/coefs[0])])
        self.x_slice_InfLineLabel.setText(label_text)
        self.fit_line.setData(x,fit)
        
    def poly_fit_map(self, time_array, time_trace_map):
        coefs_map = poly_fit_map(time_array=time_array-time_array.min(), time_trace_map=time_trace_map)
        print(coefs_map.shape)
        return -1/coefs_map[:,:,0]

def poly_fit(y,x,deg=1):
        mask = y > 0
        x = x[mask]
        y = y[mask]
        t = x.copy()
        t -= t.min()
        coefs = np.polyfit(t,np.log(y),deg)
        return coefs
def poly_fit_map(time_array, time_trace_map, axis=-1):
    kwargs = dict(x=time_array)
    return np.apply_along_axis(poly_fit, axis=axis, arr=np.squeeze(time_trace_map), **kwargs)

def tau_x_calc(time_trace, time_array, x=0.6321205588300001):
    t = time_trace
    return time_array[np.argmin(np.abs(np.cumsum(t)/np.sum(t)-x))]        
def tau_x_calc_map(time_array, time_trace_map, x=0.6321205588300001, axis=-1):
    kwargs = dict(time_array=time_array, x=x)
    return np.apply_along_axis(tau_x_calc, axis=axis, arr=np.squeeze(time_trace_map), **kwargs)

"""class TRPL3dNPZView(HyperSpectralBaseView):

    name = 'trpl_3d_npz'
    
    def setup(self):
        HyperSpectralBaseView.setup(self)
        TRPLNPZView.scan_specific_setup(self)
        
        self.settings.New('plane', dtype=str, initial='xy', choices=('xy', 'yz', 'xz'))
        self.settings.New('index', dtype=int)
        self.settings.New('auto_level', dtype=bool, initial=True)
        for name in ['plane', 'index', 'auto_level']:
            self.settings.get_lq(name).add_listener(self.update_display)
        
        #self.ui = QtWidgets.QWidget()
        #self.ui.setLayout(QtWidgets.QVBoxLayout())
        self.dockarea.addDock(name='Image', widget=self.settings.New_UI())
        self.info_label = QtWidgets.QLabel()
        self.dockarea.addDock(name='info', widget=self.info_label)
        #self.imview = pg.ImageView()
        #self.ui.layout().addWidget(self.imview, stretch=1)
        
        #self.graph_layout = pg.GraphicsLayoutWidget()
        #self.graph_layout.addPlot()
        
    def on_change_data_filename(self, fname):
        
        try:
            TRPLNPZView.load_data(self, fname)
            self.update_display()
        except Exception as err:
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return "trpl_scan3d.npz" in fname
    
    def update_display(self):
        
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
        
        HyperSpectralBaseView.update_display(self)"""

        

if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(TRPLNPZView(app))
    
    sys.exit(app.exec_())