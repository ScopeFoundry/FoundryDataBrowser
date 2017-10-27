from ScopeFoundry.data_browser import DataBrowserView
import pyqtgraph as pg
import h5py
from qtpy import QtWidgets
import numpy as np

class PowerScanH5View(DataBrowserView):
    
    name = 'power_scan_h5'
    
    def is_file_supported(self, fname):
        return('power_scan' in fname) and ('.h5' in fname)
            
    def setup(self):
        
        self.settings.New('spec_index', dtype=int, initial=0)
        self.settings.spec_index.add_listener(self.on_spec_index_change)
        
        self.settings.New("x_axis", dtype=str, initial='power_wheel', choices=('power_wheel', 'pm_power'))
        
        self.ui = QtWidgets.QGroupBox()
        self.ui.setLayout(QtWidgets.QVBoxLayout())
        
        self.ui.spec_index_doubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.settings.spec_index.connect_bidir_to_widget(self.ui.spec_index_doubleSpinBox)
        self.ui.layout().addWidget(self.ui.spec_index_doubleSpinBox)
        
        self.ui.x_axis_comboBox = QtWidgets.QComboBox()
        self.settings.x_axis.connect_to_widget(self.ui.x_axis_comboBox)
        self.ui.layout().addWidget(self.ui.x_axis_comboBox)
        
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.ui.layout().addWidget(self.graph_layout)
        
        self.power_plot = self.graph_layout.addPlot()
        self.power_plot.setLogMode(x=True, y=True)
        
        self.power_plotcurve = self.power_plot.plot([1],[1])
        
        self.power_plot_current_pos = self.power_plot.plot(symbol='o')
        #self.power_plot_arrow = pg.ArrowItem()
        #self.power_plot_arrow.setPos(0,0)
        #self.power_plot.addItem(self.power_plot_arrow)
        
        self.graph_layout.nextRow()
        
        self.spec_plot = self.graph_layout.addPlot()
        self.spec_plotcurve = self.spec_plot.plot([0])
        
        
    def on_change_data_filename(self, fname=None):

        try:        
            self.dat = h5py.File(fname, 'r')
        
            if 'measurement/power_scan_df' in self.dat:
                self.H = self.dat['measurement/power_scan_df']
            else:
                self.H = self.dat['measurement/power_scan']
                
            H = self.H
            
            self.settings.spec_index.change_min_max(0, len(H['pm_powers'])-1)
            
            self.on_change_x_axis()
            
            if 'integrated_spectra' in H:
                self.power_plot_y = np.array(H['integrated_spectra'])
                # to fix issues with log-log plotting, we shift negative data
                if np.any(self.power_plot_y < 0): 
                    self.power_plot_y -= np.min(self.power_plot_y) - 1
            elif 'picoharp_histograms' in H:
                print('picoharp')
                self.picoharp_histograms = np.array(H['picoharp_histograms'], dtype=float)
                self.picoharp_elapsed_time = np.array(H['picoharp_elapsed_time'], dtype=float)
                self.picoharp_time_array = np.array(H['picoharp_time_array'])
                self.power_plot_y = self.picoharp_histograms.sum(axis=1)/self.picoharp_elapsed_time
                if np.any(self.power_plot_y < 0): 
                    self.power_plot_y -= np.min(self.power_plot_y) - 1
            else:
                self.power_plot_y = np.array(H['pm_powers'],dtype=float)
                
            self.power_plotcurve.setData(self.X, self.power_plot_y)
            self.settings['spec_index'] = 0
            self.on_spec_index_change()
            self.databrowser.ui.statusbar.showMessage("loaded %s" %(fname))

        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
    
    def on_spec_index_change(self):
        ii = self.settings['spec_index']
        print("on_spec_index_change", ii)
        print(list(self.H.keys()))
        H = self.H
        
        self.power_plot_current_pos.setData(self.X[ii:ii+1], self.power_plot_y[ii:ii+1])
        
        #if 'integrated_spectra' in H:
        #    print(H['pm_powers'][ii], H['integrated_spectra'][ii])
        #    self.power_plot_arrow.setPos(np.log10(H['pm_powers'][ii]), np.log10(self.power_plot_y[ii]))
            
        if 'spectra' in H:
            self.spec_plotcurve.setData(H['wls'], H['spectra'][ii])
        elif 'picoharp_histograms' in H:
            print("ASdf")
            self.spec_plotcurve.setData(self.picoharp_time_array, self.picoharp_histograms[ii,:])  
        else:
            self.spec_plotcurve.setData([0])
        #self.power_plot_arrow.setPos(np.log10(H['pm_powers'][ii]), np.log10(self.power_plot_y[ii]))
        
    def on_change_x_axis(self):
        if self.settings['x_axis'] == 'power_wheel':    
            self.X = np.array(self.H['power_wheel_position'], dtype=float)
        else:
            self.X = np.array(self.H['pm_powers'], dtype=float)
        #self.on_spec_index_change()

        