from ScopeFoundry.data_browser import HyperSpectralBaseView
import numpy as np

class Picoharp_MCL_2DSlowScan_View(HyperSpectralBaseView):
    
    name = 'Picoharp_MCL_2DSlowScan_View'

    def is_file_supported(self, fname):
        return "Picoharp_MCL_2DSlowScan.h5" in fname
    
    def scan_specific_setup(self):
        # set spectral plot to be semilog-y
        self.spec_plot.setLogMode(False, True)
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'Time', units='ns')

    
    def load_data(self, fname):
        # return hyperspec data, display image
        import h5py
        
        self.dat = h5py.File(fname, 'r')
        self.time_trace_map = np.array(self.dat['/measurement/Picoharp_MCL_2DSlowScan/time_trace_map'])
        self.time_array = np.array(self.dat['measurement/Picoharp_MCL_2DSlowScan/time_array'])

        self.hyperspec_data = self.time_trace_map[0]+1
        self.display_image = self.time_trace_map[0,:,:,:].sum(axis=2)
        self.spec_x_array = self.time_array
        
class FiberPicoharpScanView(HyperSpectralBaseView):
    
    name = 'fiber_picoharp_scan'

    def is_file_supported(self, fname):
        return "fiber_picoharp_scan.h5" in fname
    
    def scan_specific_setup(self):
        # set spectral plot to be semilog-y
        self.spec_plot.setLogMode(False, True)
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'Time', units='ns')

    
    def load_data(self, fname):
        # return hyperspec data, display image
        import h5py
        
        self.dat = h5py.File(fname, 'r')
        self.time_trace_map = np.array(self.dat['/measurement/fiber_picoharp_scan/time_trace_map'])
        self.time_array = np.array(self.dat['measurement/fiber_picoharp_scan/time_array'])

        self.hyperspec_data = self.time_trace_map[0]+1
        self.display_image = self.time_trace_map[0,:,:,:].sum(axis=2)
        self.spec_x_array = self.time_array
    