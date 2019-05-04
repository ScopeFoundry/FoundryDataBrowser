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

        self.power_plot_current_pos = self.power_plot.plot(symbol='o', pen='r')

        self.power_fit_plotcurve = self.power_plot.plot([1],[1],pen='r', name='Fit')

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

        settings_layout.addWidget(QtWidgets.QLabel('power x axis:'),1,0)
        self.ui.power_x_axis_comboBox = QtWidgets.QComboBox()
        self.settings.power_x_axis.connect_to_widget(self.ui.power_x_axis_comboBox)
        settings_layout.addWidget(self.ui.power_x_axis_comboBox,1,1)

        
        self.power_plot_slicer = RegionSlicer(self.power_plotcurve, name='fit slicer',
                                              slicer_updated_func=self.redo_fit,
                                              activated = True,
                                              )        
        settings_layout.addWidget(self.power_plot_slicer.New_UI(),2,0)

        self.spec_x_slicer = RegionSlicer(self.spec_plotcurve, name='slicer',
                                     slicer_updated_func=self.update_power_plotcurve,
                                     activated = True,
                                    )
        settings_layout.addWidget(self.spec_x_slicer.New_UI(),2,1)
    
        
    def on_change_data_filename(self, fname=None):

        try:        
            self.h5file = h5py.File(fname, 'r')
            
            if 'measurement/power_scan_df' in self.h5file:
                self.H = self.h5file['measurement/power_scan_df']
            else:
                self.H = self.h5file['measurement/power_scan']
                
            H = self.H
            
            self.settings.spec_index.change_min_max(0, len(H['pm_powers'])-1)
                                    
            if 'integrated_spectra' in H:
                self.spec_x_array = np.array(H['wls']) 
                self.hyperspec_data = np.array(H['spectra'])
                    
            elif 'picoharp_histograms' in H:
                self.picoharp_histograms = np.array(H['picoharp_histograms'], dtype=float)
                self.picoharp_elapsed_time = np.array(H['picoharp_elapsed_time'], dtype=float)
                self.spec_x_array = np.array(H['picoharp_time_array'])                   
                self.hyperspec_data = (self.picoharp_histograms.T/self.picoharp_elapsed_time).T
            else:
                self.hyperspec_data = 0.5*np.ones((len(H['pm_powers'][:]),2))*H['pm_powers'][:]
                self.spec_x_array = np.array([0,1])
            # get x-axis values
            for key in self.power_x_axis_choices:
                try:              
                    setattr(self, key, np.array(H[key], dtype=float))
                except:
                    pass
                    
            self.h5file.close()
            
            self.update_power_plotcurve()
            self.settings['spec_index'] = 0
            self.on_spec_index_change()
            
            self.on_change_power_x_axis()

            self.spec_plotcurve_mean.setData(self.spec_x_array, self.hyperspec_data.mean(axis=0))
            
            self.databrowser.ui.statusbar.showMessage("loaded:{}\n".format(fname))

        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load {}:\n{}".format(fname, err) )
            raise(err)
    
    def on_spec_index_change(self):
        ii = self.settings['spec_index']
        self.power_plot_current_pos.setData(self.X[ii:ii+1], self.Y[ii:ii+1])
        spectrum = self.hyperspec_data[self.idx_mapping[ii],:]
        self.spec_plotcurve.setData(self.spec_x_array,spectrum)
        
    def on_change_power_x_axis(self):
        self.update_power_plotcurve()
    
    def get_power_xhyperspecdata(self):
        power_plot_x = getattr(self, self.settings.power_x_axis.value)
        self.idx_mapping = idx = np.argsort(power_plot_x)
        hyperspec = self.hyperspec_data[idx,self.spec_x_slicer.s]
        return (power_plot_x[idx],hyperspec)
        
    def get_power_xy(self):
        power_plot_x,hyperspec = self.get_power_xhyperspecdata()
        power_plot_y  = hyperspec.sum(axis=1)
        if np.any(power_plot_y < 0): 
            power_plot_y -= np.min(power_plot_y) - 1        
        return (power_plot_x,power_plot_y)
        
    def update_power_plotcurve(self):      
        self.X, self.Y = self.get_power_xy() 
        self.power_plotcurve.setData(self.X, self.Y)
        self.on_spec_index_change()

    def redo_fit(self):
        s = self.power_plot_slicer.s
        m, b = np.polyfit(np.log10(self.X[s]), np.log10(self.Y[s]), deg=1)
        print("fit values m,b:", m,b) 
        fit_data = 10**(np.poly1d((m,b))(np.log10(self.X)))
        self.power_fit_plotcurve.setData(self.X[s], fit_data[s])
        self.power_plot_slicer.set_label("<h1>{:1.2f}+I<sup>{:1.2f}</sup></h1>".format(b,m))

        