from ScopeFoundry.data_browser import DataBrowser, HyperSpectralBaseView
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets


def t_x_calc(time_array, time_trace_map, kk_start, kk_stop, x=1-0.36787944117, bgsub=True):

    kk_bg_max = int(4*kk_start/5)

    bg_slice = slice(0,kk_bg_max) #fit_config.bg_slice #slice(0,kk_bg_max/2)

    if len(time_trace_map.shape) == 4: #if 4d (3d + time) data
        Nz, Ny, Nx, Nt = time_trace_map.shape 
        bg = np.average(time_trace_map[:,:,:,bg_slice], axis=3).reshape(Nz,Ny,Nx,1)
        T = np.array(time_trace_map[:,:,:,kk_start:kk_stop], dtype=float) # copy array
        if bgsub:
            T -= bg

        t_x_map = time_array[  np.argmin(
                               np.abs( np.cumsum(T, axis=3)/ 
                                          np.sum(T, axis=3).reshape(Nz, Ny, Nx,1)
                                          - x), axis=3)]
    else: #if 3d (2d + time) data
        Ny, Nx, Nt = time_trace_map.shape
        bg = np.mean(time_trace_map[:,:,bg_slice], axis=2).reshape(Ny, Nx,1)
        
        T = np.array(time_trace_map[:,:,kk_start:kk_stop], dtype=float) # copy array
        if bgsub:
            T -= bg

        t_x_map =  time_array[np.argmin(
                                np.abs(np.cumsum(T, axis=2)/
                                                     np.sum(T, axis=2).reshape(Ny, Nx,1) 
                                                         - x ), axis=2)]
        
    return t_x_map, bg

class TRPL_t_x_lifetime_NPZView(HyperSpectralBaseView):

    name = 'trpl_t_x_lifetime_npz'
    
    def is_file_supported(self, fname):
        return "_trpl_scan.npz" in fname

    def load_data(self, fname):
        self.dat = np.load(fname)
        
        cr0 = self.dat['picoharp_count_rate0']
        rep_period_s = 1.0/cr0
        time_bin_resolution = self.dat['picoharp_Resolution']*1e-12
        self.num_hist_chans = int(np.ceil(rep_period_s/time_bin_resolution))
        
        # truncate data to only show the time period associated with rep-rate of laser
        
        self.time_trace_map = self.dat['time_trace_map']
        self.integrated_count_map = self.dat['integrated_count_map']
        self.time_array = self.dat['time_array']

        self.hyperspec_data = self.time_trace_map[:,:,0:self.num_hist_chans]+1
        self.display_image = self.integrated_count_map
        self.spec_x_array = self.time_array[0:self.num_hist_chans]

        self.compute_lifetime_map()
    
    def scan_specific_setup(self):
        
        self.settings.New('kk_start', dtype=int, initial=0)
        self.settings.New('kk_stop',  dtype=int, initial=100)
        self.settings.New('bg_sub',   dtype=bool, initial=True)
        self.settings.New('e_exp', dtype=float, initial=1.0)
        self.settings.New('auto_recompute', dtype=bool, initial=True)
        
        self.settings_ui = self.settings.New_UI()
        self.compute_button = QtWidgets.QPushButton("Go")
        self.settings_ui.layout().addRow("Compute:", self.compute_button)
        self.compute_button.clicked.connect(self.compute_lifetime_map)
        
        self.splitter.insertWidget(0, self.settings_ui )
        
        self.settings.kk_start.add_listener(self.on_update_kk_bounds)
        self.settings.kk_stop.add_listener(self.on_update_kk_bounds) 
        
        for lqname in ['kk_start', 'kk_stop', 'bg_sub', 'e_exp']:
            self.settings.get_lq(lqname).add_listener(self.on_param_changes)
               
            
        # set spectral plot to be semilog-y
        self.spec_plot.setLogMode(False, True)
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'time', units='ns')
        
        self.kk_start_vline = pg.InfiniteLine(0, angle=90, pen=1, movable=False, name='kk_start')
        self.kk_stop_vline = pg.InfiniteLine(0, angle=90, pen=1, movable=False, name='kk_stop')
        self.tau_x_vline = pg.InfiniteLine(0, angle=90, pen=1, movable=False, name='tau_x')
    
        self.spec_plot.addItem(self.kk_start_vline,  ignoreBounds=True)
        self.spec_plot.addItem(self.kk_stop_vline,  ignoreBounds=True)
        self.spec_plot.addItem(self.tau_x_vline,  ignoreBounds=True)
        
        self.point_plotdata_bgsub = self.spec_plot.plot(pen='g')
        

    def on_update_circ_roi(self, roi=None):
        HyperSpectralBaseView.on_update_circ_roi(self, roi=roi)

        j,i = self.circ_roi_ji
        self.point_plotdata_bgsub.setData(self.time_array[0:self.num_hist_chans],
                                          self.time_trace_map_bgsub[j,i,0:self.num_hist_chans]+1)
        self.tau_x_vline.setPos(self.time_array[self.settings['kk_start']] + self.tau_x_map[j,i])
        
    def on_update_kk_bounds(self):
        self.kk_start_vline.setPos(self.time_array[self.settings['kk_start']])
        self.kk_stop_vline.setPos( self.time_array[self.settings['kk_stop' ]])
        
    def on_param_changes(self):
        if self.settings['auto_recompute']:
            self.compute_lifetime_map()
        
    def compute_lifetime_map(self):
        self.tau_x_map, self.bg = t_x_calc(self.time_array, 
                                           self.time_trace_map,
                                           x = 1 - np.exp(-1*self.settings['e_exp']),
                                           kk_start=self.settings['kk_start'], 
                                           kk_stop=self.settings['kk_stop'],
                                           bgsub=self.settings['bg_sub'])
        
        self.time_trace_map_bgsub = self.time_trace_map - self.bg
        
        self.display_image = self.tau_x_map
    
        self.update_display()


if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    v = app.load_view(TRPL_t_x_lifetime_NPZView(app))
    
    app.settings['data_filename'] = "/Users/esbarnard/Dropbox/MolecularFoundry/NREL_CdTe/170314-cdte-thick/2425-R1/1489518631_trpl_scan.npz"
    
    v.settings['kk_start'] = 250
    v.settings['kk_stop'] = 750
    
    sys.exit(app.exec_())