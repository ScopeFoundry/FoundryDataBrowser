from ScopeFoundry.data_browser import DataBrowser, HyperSpectralBaseView
import numpy as np
import h5py

class WinSpecMCL2DSlowScanView(HyperSpectralBaseView):

    name = 'WinSpecMCL2DSlowScan'
    
    def is_file_supported(self, fname):
        return "WinSpecMCL2DSlowScan.h5" in fname   
    
    
    def load_data(self, fname):    
        self.dat = h5py.File(fname, 'r')
        self.spec_map = np.squeeze(np.array(self.dat['/measurement/WinSpecMCL2DSlowScan/spec_map']), axis=(3,4))
        
        self.integrated_count_map =  self.spec_map.sum(axis=3)

        self.hyperspec_data = self.spec_map[0] # pick frame 0
        self.display_image = self.integrated_count_map[0]
        self.spec_x_array = self.dat['/measurement/WinSpecMCL2DSlowScan/wavelength']
        
    def scan_specific_setup(self):
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'Wavelength', units='nm')        
        
