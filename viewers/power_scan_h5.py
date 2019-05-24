from ScopeFoundry.data_browser import DataBrowserView
import pyqtgraph as pg
import h5py
from qtpy import QtWidgets
import numpy as np
from ScopeFoundry.widgets import RegionSlicer


class PowerScanH5View(DataBrowserView):
    
    name = 'power_scan_h5'
    
    def is_file_supported(self, fname):
        return('power_scan' in fname) and ('.h5' in fname)
            
    def setup(self):
        self.settings.New('spec_index', dtype=int, initial=0)
        self.settings.spec_index.add_listener(self.on_spec_index_change)
        
        self.settings.New('chan', dtype=int, initial=0)
        self.settings.chan.add_listener(self.update_power_plotcurve)

        self.power_x_axis_choices = ('pm_powers', 'pm_powers_after', 'power_wheel_position')
        self.settings.New("power_x_axis", dtype=str, initial='pm_powers', choices=self.power_x_axis_choices)
        self.settings.power_x_axis.add_listener(self.on_change_power_x_axis)
        
        self.ui = QtWidgets.QGroupBox()
        self.ui.setLayout(QtWidgets.QVBoxLayout())
        
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.ui.layout().addWidget(self.graph_layout)

        self.power_plot = self.graph_layout.addPlot()
        self.power_plot.setLogMode(x=True, y=True)
        self.power_plotcurve = self.power_plot.plot([1],[1], name='Data', symbol='+', symbolBrush='m')
        self.power_plot_current_pos = self.power_plot.plot(symbol='o', symbolBrush='r',)
        self.power_plot_current_pos.setZValue(10)
        self.power_fit_plotcurve = self.power_plot.plot([1],[1],pen='g', name='Fit')
        self.power_plotcurve_selected = self.power_plot.plot([1],[1],symbol='o', pen=None, symbolPen='g') 

        self.graph_layout.nextRow()
        self.spec_plot = self.graph_layout.addPlot()
        self.spec_plotcurve = self.spec_plot.plot([0], pen='r' )
        self.spec_plotcurve_mean = self.spec_plot.plot([0], name='mean')

        settings_layout = QtWidgets.QGridLayout()
        self.ui.layout().addLayout(settings_layout)
        
        settings_layout.addWidget(QtWidgets.QLabel('data index:'),0,0)
        self.ui.spec_index_doubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.settings.spec_index.connect_to_widget(self.ui.spec_index_doubleSpinBox)
        settings_layout.addWidget(self.ui.spec_index_doubleSpinBox,0,1)

        settings_layout.addWidget(QtWidgets.QLabel('data channel:'),1,0)
        self.ui.chan_doubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.settings.chan.connect_to_widget(self.ui.chan_doubleSpinBox)
        settings_layout.addWidget(self.ui.chan_doubleSpinBox,1,1)

        settings_layout.addWidget(QtWidgets.QLabel('power x-axis:'),2,0)
        self.ui.power_x_axis_comboBox = QtWidgets.QComboBox()
        self.settings.power_x_axis.connect_to_widget(self.ui.power_x_axis_comboBox)
        settings_layout.addWidget(self.ui.power_x_axis_comboBox,2,1)

        
        self.power_plot_slicer = RegionSlicer(self.power_plotcurve, name='fit slicer',
                                              slicer_updated_func=self.redo_fit,
                                              activated = True,
                                              )        
        settings_layout.addWidget(self.power_plot_slicer.New_UI(),3,0)

        self.spec_x_slicer = RegionSlicer(self.spec_plotcurve, name='spec_x_slicer',
                                     slicer_updated_func=self.update_power_plotcurve,
                                     activated = False,
                                    )
        settings_layout.addWidget(self.spec_x_slicer.New_UI(),3,1)
        
        self.bg_slicer = RegionSlicer(self.spec_plotcurve, name='bg subtract',
                                     slicer_updated_func=self.update_power_plotcurve,
                                     activated = False,
                                    )
        settings_layout.addWidget(self.bg_slicer.New_UI(),3,2)
        
    
        
    def on_change_data_filename(self, fname=None):

        try:        
            self.h5file = h5py.File(fname, 'r')
            
            if 'measurement/power_scan_df' in self.h5file:
                self.H = self.h5file['measurement/power_scan_df']
            else:
                self.H = self.h5file['measurement/power_scan']
                
            H = self.H
            
            # get power arrays
            self.power_arrays = {}
            for key in self.power_x_axis_choices:
                try:              
                    self.power_arrays.update({key:H[key][:]})
                except:
                    pass
            
            
            Np = len(self.power_arrays['pm_powers'])
            self.settings.spec_index.change_min_max(0, Np-1)          
              
            #Provide  spec_x_array and hyperspec_data for each.
            '''self.spec_x_array has shape (N_wls,) [dim=1]
            self.hyperspec_data has shape (Np, N_channels, N_wls)  [dim=3]'''
            self.spec_x_array = np.arange(512) 
            self.hyperspec_data = 0.5*np.arange(512*Np*1).reshape((Np, 1, 512))

                                    
            if 'integrated_spectra' in H:
                self.spec_x_array = H['wls'][:] 
                self.hyperspec_data = H['spectra'][:].reshape(Np,1,-1)
                    
            for harp in ['picoharp','hydraharp']:
                if '{}_histograms'.format(harp) in H:
                    histograms = H['{}_histograms'.format(harp)][:]
                    elapsed_time = H['{}_elapsed_time'.format(harp)][:]
                    self.spec_x_array = H['{}_time_array'.format(harp)][:]
                    if np.ndim(histograms) == 2:
                        histograms = histograms.reshape(Np,1,-1)
                    self.hyperspec_data = (histograms.T/elapsed_time).T
                    
            self.h5file.close()
            
            self.update_power_plotcurve()
            self.settings['spec_index'] = 0
            self.on_spec_index_change()
            
            self.on_change_power_x_axis()

            self.databrowser.ui.statusbar.showMessage("loaded:{}\n".format(fname))
            
            n_chan = self.hyperspec_data.shape[1]
            self.settings.chan.change_min_max(0, n_chan-1)
            self.ui.chan_doubleSpinBox.setEnabled(bool(n_chan-1))
            

        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load {}:\n{}".format(fname, err) )
            raise(err)
    
    
    def on_spec_index_change(self):
        ii = self.settings['spec_index']
        self.power_plot_current_pos.setData(self.X[ii:ii+1], self.Y[ii:ii+1])
        _,spectra = self.get_power_xhyperspecdata(apply_x_slicer=False)
        print(spectra.shape)
        self.spec_plotcurve.setData(self.spec_x_array,spectra[ii,:])
        
        #show power wheel position
        power_wheel_position = self.power_arrays['power_wheel_position'][ii]
        self.databrowser.ui.statusbar.showMessage("power_wheel_position: {:1.1f}".format(power_wheel_position))
        self.spec_plot.setTitle("power_wheel_position: {:1.1f}".format(power_wheel_position), color='r')
        
    def on_change_power_x_axis(self):
        self.update_power_plotcurve()
        
    def get_bg(self):
        if self.bg_slicer.activated.val:
            bg = self.hyperspec_data[:,self.settings['chan'],self.bg_slicer.slice].mean()
        else:
            bg = 0
        return bg    
 
    def get_power_xhyperspecdata(self, apply_x_slicer=True):
        bg = self.get_bg()
        power_plot_x = self.power_arrays[self.settings['power_x_axis']]
        hyperspec_data = self.hyperspec_data[:,self.settings['chan'],:]
        if apply_x_slicer:
            hyperspec_data = hyperspec_data[:,self.spec_x_slicer.s_]
        return (power_plot_x, hyperspec_data-bg)
                
    def get_power_xy(self):
        power_plot_x,hyperspec = self.get_power_xhyperspecdata(apply_x_slicer=True)
        power_plot_y  = hyperspec.sum(axis=1)
        if np.any(power_plot_y < 0): 
            power_plot_y -= np.min(power_plot_y) - 1        
        return (power_plot_x,power_plot_y)
        
    def update_power_plotcurve(self):      
        self.X, self.Y = self.get_power_xy() 
        self.power_plotcurve.setData(self.X, self.Y)
        self.on_spec_index_change()
        self.redo_fit()

    def redo_fit(self):
        s = self.power_plot_slicer.mask
        m, b = np.polyfit(np.log10(self.X[s]), np.log10(self.Y[s]), deg=1)
        print("fit values m,b:", m,b) 
        fit_data = 10**(np.poly1d((m,b))(np.log10(self.X)))
        self.power_fit_plotcurve.setData(self.X[s], fit_data[s])
        self.power_plotcurve_selected.setData(self.X[s], self.Y[s])
        self.power_plot_slicer.set_label("<h1>{:1.2f} * I<sup>{:1.2f}</sup></h1>".format(b,m))      
        
        
        
if __name__ == '__main__':
    import sys
    from ScopeFoundry.data_browser import DataBrowser
    app = DataBrowser(sys.argv)

    
    from FoundryDataBrowser.viewers.h5_tree import H5TreeSearchView
    app.load_view(H5TreeSearchView(app))    
    app.load_view(PowerScanH5View(app))
    
    sys.exit(app.exec_()) 