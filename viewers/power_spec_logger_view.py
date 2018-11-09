from ScopeFoundry.data_browser import DataBrowserView, DataBrowser
import pyqtgraph as pg
import h5py
import numpy as np


class PowerSpectrumLoggerView(DataBrowserView):
    
    name = 'power_spec_logger'
    
    def setup(self):
        
        self.ui = self.graph_layout = pg.GraphicsLayoutWidget()

        p = self.power_spec_plot = self.graph_layout.addPlot(row=0, col=0)
        self.dc_plotlines = [None,None,None]

        p.setTitle("Power Spectrum")
        p.addLegend()

        p.current_plotline = p.plot(pen=pg.mkPen('r'), name='Current')
        p.avg_plotline = p.plot(name='Running average')
    
        #p.showLabel('top', True)
        p.setLabel('bottom', "Frequency", 'Hz')
        p.setLabel('left', 'PSD [V<sup>2</sup>/Hz]')
        p.setLogMode(x=False, y=True)
        
        #self.settings.view_freq_min.add_listener(self.on_update_freq_lims)
        #self.settings.view_freq_max.add_listener(self.on_update_freq_lims)
        
        #self.on_update_freq_lims()


                    
        dc_p = self.dc_plot = self.graph_layout.addPlot(row=1, col=0)
        dc_p.addLegend()
        dc_p.setTitle("DC")
        dc_p.setLabel('bottom', "Time", 's')
        dc_p.setLabel('left', '&Delta; <sub>DC</sub>', units='V')
        
        dc_p.addItem(pg.InfiniteLine(movable=False, angle=0))

        for i,name in enumerate('xyz'):
            self.dc_plotlines[i] = dc_p.plot(pen=pg.mkPen('rgb'[i]), name=name, autoDownsampleFactor=1.0)
        
        dc_p.setDownsampling(auto=True, mode='subsample',)
        
        
        p = self.roi_plot = self.graph_layout.addPlot(row=2, col=0)
        p.addLegend()
        p.setTitle("Frequency Band History")
        p.setLabel('bottom', "Time", 's')        
        p.setXLink(self.dc_plot)
        p.setLabel('left', 'V')
        p.setLogMode(x=False, y=True)
        
        self.linear_range_items = []
        for i in range(2):
            color = ['#F005','#0F05'][i]
            lr = pg.LinearRegionItem(values=[55*(i+1),65*(i+1)], brush=pg.mkBrush(color))
            lr.num = i
            self.power_spec_plot.addItem(lr)
            lr.label = pg.InfLineLabel(lr.lines[0], "Region {}".format(i), position=0.8, rotateAxis=(1,0), anchor=(1, 1), movable=True)
            self.linear_range_items.append(lr)
            
            lr.hist_plotline = self.roi_plot.plot(pen=pg.mkPen(color[:-1]), name='Region {}'.format(i))
            lr.hist_plotline.setAlpha(1, auto=False)

            ### add listenr for linear_range_items changing range
            lr.sigRegionChanged.connect(self.update_display)
        
        self.roi_plot.setDownsampling(auto=True, mode='subsample',)

        #self.settings.phys_unit.add_listener(self.on_change_phys_unit)
        

    def is_file_supported(self, fname):
        for m_name in ['accel_logger', 'mag_logger', 'power_spec_logger']:                 
            if m_name in fname:
                return True
            
        return False

    def on_change_data_filename(self, fname=None):
        dat = self.dat = h5py.File(fname, 'r')
        for m_name in ['accel_logger', 'mag_logger', 'power_spec_logger']:                 
            if m_name in dat['measurement/']:
                M = self.M = dat['measurement/' + m_name]
                
                
        ### copy data
        self.dc_time = np.array(M['time'])
        self.dc_history = np.array(M['dc_history'])
        self.psd_freq = np.array(M['psd_freq'])
        self.psd_history = np.array(M['psd_history'])
        
        self.mean_dc = self.dc_history.mean(axis=0) # average over time
        self.mean_psd = self.psd_history.mean(axis=0) # average over time
        
        self.n_chans = self.dc_history.shape[1]
        
        self.settings_dict = dict( M['settings'].attrs )
        
        ### update display
        self.on_change_phys_unit()
        self.update_display()
        
    def on_change_phys_unit(self):
        #unit = self.settings['phys_unit']
        unit = self.settings_dict['phys_unit']
        self.power_spec_plot.setLabel('left', 'PSD [{}<sup>2</sup>/Hz]'.format(unit))
        self.dc_plot.setLabel('left', '&Delta; <sub>DC</sub>', units=unit)
        self.roi_plot.setLabel('left', unit)
        #self.settings.scale.change_unit("{}/V".format(unit))


    def update_display(self):
        self.dc_plot.setVisible(self.settings_dict['view_show_dc'])

        #unit = self.settings['phys_unit']
        unit = self.settings_dict['phys_unit']

        #self.power_spec_plot.current_plotline.setData(self.psd_freq[:], self.current_psd[:,:].sum(axis=1))
        self.power_spec_plot.avg_plotline.setData(self.psd_freq[:], self.mean_psd[:,:].sum(axis=1))
    
        #downsample_int = 1000
        #N = self.N
        # need to find N
        N = np.argmax(self.dc_time)
        

        
        for i in range(self.n_chans):
            self.dc_plotlines[i].setData(self.dc_time[:N], self.dc_history[:N,i]-self.mean_dc[i])
            
        if self.n_chans == 3:
            x,y,z = self.mean_dc
            self.dc_plot.setTitle("DC x={:1.2f} {}   y={:1.2f} {}   z={:1.2f} {}".format(x, unit, y,unit, z,unit))
        else:
            self.dc_plot.setTitle("DC {} {}".format(self.mean_dc, unit))
        
        
        
        for lr in self.linear_range_items:
            f0, f1 = lr.getRegion()
            
            new_label = "R{}: {:1.1f} Hz : {:1.1f}".format(lr.num, 0.5*(f0+f1), 0.5*(f1-f0))
            lr.label.setFormat(new_label)
            
            i0 = np.searchsorted(self.psd_freq, f0)
            i1 = np.searchsorted(self.psd_freq, f1)
            
            lr.hist_plotline.setData(self.dc_time[:N], np.sqrt(self.psd_history[:N, i0:i1, :].sum(axis=(1,2) ) ))
            self.roi_plot.legend.items[lr.num][1].setText(new_label)
            #lr.hist_plotline.setName("asdf")

if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(PowerSpectrumLoggerView(app))
    
    sys.exit(app.exec_())