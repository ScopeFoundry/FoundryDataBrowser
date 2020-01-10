from ScopeFoundry.data_browser import DataBrowserView
import pyqtgraph as pg
from qtpy import QtWidgets
import numpy as np

class PowerScanNPZView(DataBrowserView):
    
    name = 'power_scan_npz'
    
    def is_file_supported(self, fname):
        return('power_scan' in fname) and ('.npz' in fname)
            
    def setup(self):
        
        self.settings.New('spec_index', dtype=int, initial=0)
        
        self.ui = QtWidgets.QGroupBox()
        self.ui.setLayout(QtWidgets.QVBoxLayout())
        
        self.ui.spec_index_doubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.settings.spec_index.connect_bidir_to_widget(self.ui.spec_index_doubleSpinBox)
        self.ui.layout().addWidget(self.ui.spec_index_doubleSpinBox)
        
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.ui.layout().addWidget(self.graph_layout)
        
        self.power_plot = self.graph_layout.addPlot()
        self.power_plot.setLogMode(x=True, y=True)
        
        self.power_plotcurve = self.power_plot.plot([1],[1], name='Data')
        
        self.power_fit_plotcurve = self.power_plot.plot([1],[1],pen='r', name='Fit')
        
        self.power_plot_arrow = pg.ArrowItem()
        self.power_plot_arrow.setPos(0,0)
        self.power_plot.addItem(self.power_plot_arrow)
        
        self.power_plot_lr = pg.LinearRegionItem([1,2])
        self.power_plot_lr.setZValue(-10)
        self.power_plot.addItem(self.power_plot_lr)
        self.power_plot_lr.sigRegionChanged.connect(self.redo_fit)

        #self.power_plot_legend = pg.LegendItem()
        #self.power_plot.addItem(self.power_plot_legend)
        #self.power_plot_legend.addItem(self.power_plotcurve)
        #self.power_plot_legend.addItem(self.power_fit_plotcurve)
        self.fit_text = pg.TextItem("fit")
        self.fit_text.setParentItem(self.power_plot_lr, )
        
        self.graph_layout.nextRow()
        
        self.spec_plot = self.graph_layout.addPlot()
        self.spec_plotcurve = self.spec_plot.plot([0])
        self.settings.spec_index.add_listener(self.on_spec_index_change)
        
        
        
    def on_change_data_filename(self, fname=None):
        if fname == "0":
            return

        try:        
            dat = self.dat = np.load(fname)
        
            
            self.settings.spec_index.change_min_max(0, len(dat['power_meter_power'])-1)
            
            
            if 'time_traces' in self.dat:
                self.data_avail = True
                self.power_plot_y = np.sum(dat['time_traces'], axis=1)
                
                self.power_plot.setLabel('left', 'Total Intensity', units='counts')
                self.power_plot.setLabel('bottom', 'Power', units='W')
                
                
                cr0 = self.dat['picoharp_count_rate0']
                rep_period_s = 1.0/cr0
                time_bin_resolution = self.dat['picoharp_Resolution']*1e-12
                self.num_hist_chans = int(np.ceil(rep_period_s/time_bin_resolution))

                self.spec_plot.setLogMode(False, True)
                self.spec_plot.setLabel('left', 'Intensity', units='counts')
                self.spec_plot.setLabel('bottom', 'time', units='ns')
                
            elif 'integrated_spectra' in self.dat:
                self.data_avail = True
                self.power_plot_y = np.array(dat['integrated_spectra'])
                

            else:
                self.data_avail = False    
                self.power_plotcurve.setData(dat['power_meter_power'])


            if self.data_avail:
                # to fix issues with log-log plotting, we shift negative data
                if np.any(self.power_plot_y < 0): 
                    self.power_plot_y -= np.min(self.power_plot_y) - 1
                x = dat['power_meter_power']
                self.power_plotcurve.setData(x, self.power_plot_y)
                
                try:
                    self.redo_fit()
                except Exception as err:
                    print("failed to fit", err)

            self.settings['spec_index'] = 0
            self.on_spec_index_change()

        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
    
    def on_spec_index_change(self):
        ii = self.settings['spec_index']
        
        H = self.dat

        if 'time_traces' in self.dat:
            self.spec_plotcurve.setData(H['time_array'][:self.num_hist_chans],
                                        1+H['time_traces'][ii,:self.num_hist_chans])
            
        if 'integrated_spectra' in H:
            print(H['power_meter_power'][ii], H['integrated_spectra'][ii])
            
        if 'spectra' in H:
            self.spec_plotcurve.setData(H['wls'], H['spectra'][ii])
            
        if self.data_avail:
            self.power_plot_arrow.setPos(np.log10(H['power_meter_power'][ii]), np.log10(self.power_plot_y[ii]))
            
    def redo_fit(self):
        lx0, lx1 = self.power_plot_lr.getRegion()
        
        x0, x1 = 10**lx0, 10**lx1
        
        
        X = self.dat['power_meter_power']
        
        n = len(X)
        
        ii0 = np.argmin(np.abs(X[:n//2+1]-x0))
        ii1 = np.argmin(np.abs(X[:n//2+1]-x1))
        
        print(ii0,ii1)

        m, b = np.polyfit(np.log10(X[ii0:ii1]), np.log10(self.power_plot_y[ii0:ii1]), deg=1)
        print("fit", m,b) 

        fit_data = 10**(np.poly1d((m,b))(np.log10(X)))
        print("fit_data", fit_data)    
        self.power_fit_plotcurve.setData(X, fit_data)
        
        self.fit_text.setHtml("<h1>I<sup>{:1.2f}</sup></h1>".format(m))
        self.fit_text.setPos(0.5*(lx0+lx1), np.log10(fit_data[(ii0+ii1)//2]))