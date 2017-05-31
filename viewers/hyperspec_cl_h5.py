from ScopeFoundry.data_browser import HyperSpectralBaseView
import h5py
import numpy as np

class HyperSpecCLH5View(HyperSpectralBaseView):
    
    name = 'hyperspec_cl_h5'
    
    def is_file_supported(self, fname):
        return "_hyperspec_cl.h5" in fname
    
    def load_data(self, fname):    
        self.dat = h5py.File(fname)
        self.M = self.dat['measurement/hyperspec_cl']
        self.spec_map = np.squeeze(self.M['spec_map'])
        #self.integrated_count_map = self.dat['integrated_count_map']

        self.hyperspec_data = self.spec_map
        self.display_image = self.spec_map.sum(axis=-1)
        self.spec_x_array = np.arange(self.spec_map.shape[-1]) #self.dat['wls']
        
    def scan_specific_setup(self):
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'Wavelength', units='nm')        
