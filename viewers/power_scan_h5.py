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

        self.x_axis_choices = ('pm_powers', 'pm_powers_after', 'power_wheel_position')
        self.settings.New("x_axis", dtype=str, initial='pm_powers', choices=self.x_axis_choices)
        self.settings.x_axis.add_listener(self.on_change_x_axis)
        
        self.ui = QtWidgets.QGroupBox()
        self.ui.setLayout(QtWidgets.QVBoxLayout())
        
        self.ui.layout().addWidget(QtWidgets.QLabel('data index:'))
        self.ui.spec_index_doubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.settings.spec_index.connect_to_widget(self.ui.spec_index_doubleSpinBox)
        self.ui.layout().addWidget(self.ui.spec_index_doubleSpinBox)

        self.ui.layout().addWidget(QtWidgets.QLabel('x axis:'))
        self.ui.x_axis_comboBox = QtWidgets.QComboBox()
        self.settings.x_axis.connect_to_widget(self.ui.x_axis_comboBox)
        self.ui.layout().addWidget(self.ui.x_axis_comboBox)
        
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.ui.layout().addWidget(self.graph_layout)
        
        self.power_plot = self.graph_layout.addPlot()
        self.power_plot.setLogMode(x=True, y=True)
        
        self.power_plotcurve = self.power_plot.plot([1],[1], name='Data',symbol='+',symbolBrush='m')
        
        self.power_plot_current_pos = self.power_plot.plot(symbol='o')

        self.power_fit_plotcurve = self.power_plot.plot([1],[1],pen='r', name='Fit')

        self.power_plot_lr = pg.LinearRegionItem([1,2])
        self.power_plot_lr.setZValue(-10)
        self.power_plot.addItem(self.power_plot_lr)
        self.power_plot_lr.sigRegionChanged.connect(self.redo_fit)

        self.fit_text = pg.TextItem("fit")
        self.fit_text.setParentItem(self.power_plot_lr, )
        
        #self.power_plot_arrow = pg.ArrowItem()
        #self.power_plot_arrow.setPos(0,0)
        #self.power_plot.addItem(self.power_plot_arrow)
        
        self.graph_layout.nextRow()
        
        self.spec_plot = self.graph_layout.addPlot()
        self.spec_plotcurve = self.spec_plot.plot([0])
        
        
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
                self.power_plot_y = np.array(H['integrated_spectra'])
                # to fix issues with log-log plotting, we shift negative data
                if np.any(self.power_plot_y < 0): 
                    self.power_plot_y -= np.min(self.power_plot_y) - 1
                self.wls = np.array(H['wls']) 
                self.spectra = np.array(H['spectra'])
                    
            elif 'picoharp_histograms' in H:
                self.picoharp_histograms = np.array(H['picoharp_histograms'], dtype=float)
                self.picoharp_elapsed_time = np.array(H['picoharp_elapsed_time'], dtype=float)
                self.picoharp_time_array = np.array(H['picoharp_time_array'])
                self.power_plot_y = self.picoharp_histograms.sum(axis=1)/self.picoharp_elapsed_time
                if np.any(self.power_plot_y < 0): 
                    self.power_plot_y -= np.min(self.power_plot_y) - 1
            else:
                self.power_plot_y = np.array(H['pm_powers'],dtype=float)

            # get x-axis values
            for key in self.x_axis_choices:
                try:              
                    setattr(self, key, np.array(H[key], dtype=float))
                except:
                    pass
                    
            self.h5file.close()
            
            self.update_power_plotcurve()
            self.settings['spec_index'] = 0
            self.on_spec_index_change()
            
            self.databrowser.ui.statusbar.showMessage("loaded:{}\n".format(fname))

            
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load {}:\n{}".format(fname, err) )
            raise(err)
    
    def on_spec_index_change(self):
        ii = self.settings['spec_index']
        
        self.power_plot_current_pos.setData(self.power_plot_x[ii:ii+1], self.power_plot_y[ii:ii+1])
        
        if hasattr(self, 'wls'):
            self.spec_plotcurve.setData(self.wls, self.spectra[ii])
        elif hasattr(self, 'picoharp_time_array'):
            self.spec_plotcurve.setData(self.picoharp_time_array, self.picoharp_histograms[ii,:])  
        else:
            self.spec_plotcurve.setData([0])
        #self.power_plot_arrow.setPos(np.log10(H['pm_powers'][ii]), np.log10(self.power_plot_y[ii]))
           
        
    def update_power_plotcurve(self):      
        self.power_plot_x = getattr(self, self.settings.x_axis.value)
        self.power_plotcurve.setData(self.power_plot_x, self.power_plot_y)    
        
        idx = np.argsort(self.power_plot_x)
        self.X = self.power_plot_x[idx] 
        self.Y = self.power_plot_y[idx]
        
        
    def on_change_x_axis(self):
        self.update_power_plotcurve()
        

    def redo_fit(self):
        lx0, lx1 = self.power_plot_lr.getRegion()        
        x0, x1 = 10**lx0, 10**lx1

        X = self.X
        ii0 = np.argmin(np.abs(X-x0))
        ii1 = np.argmin(np.abs(X-x1))
        m, b = np.polyfit(np.log10(X[ii0:ii1]), np.log10(self.Y[ii0:ii1]), deg=1)
        print("fit values m,b:", m,b) 

        fit_data = 10**(np.poly1d((m,b))(np.log10(X)))       
        self.power_fit_plotcurve.setData(X, fit_data)
        
        self.fit_text.setHtml("<h1>I<sup>{:1.2f}</sup></h1>".format(m))
        self.fit_text.setPos(0.5*(lx0+lx1), np.log10(fit_data[(ii0+ii1)//2]))

        