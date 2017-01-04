from ScopeFoundry.data_browser import DataBrowser, HyperSpectralBaseView
import numpy as np

class TRPLNPZView(HyperSpectralBaseView):

    name = 'trpl_scan_npz'
    
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
    
    def scan_specific_setup(self):
        # set spectral plot to be semilog-y
        self.spec_plot.setLogMode(False, True)
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'time', units='ns')
    
        

if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(TRPLNPZView(app))
    
    sys.exit(app.exec_())