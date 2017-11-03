from ScopeFoundry.data_browser import HyperSpectralBaseView
import h5py
import numpy as np

class HyperSpecIRView(HyperSpectralBaseView):
    
    name = 'hyperspec_IR_h5'
    
    def is_file_supported(self, fname):
        return "_m4_hyperspectral_2d_scan.h5" in fname
    
    def load_data(self, fname):    
        self.dat = h5py.File(fname)
        self.spec_map = self.dat['measurement/m4_hyperspectral_2d_scan/hyperspectral_map'][0,:,:,:]
        print(self.spec_map.shape)
        #self.integrated_count_map = self.dat['integrated_count_map']

        self.hyperspec_data = self.spec_map
        self.display_image = self.spec_map.sum(axis=-1)
        self.spec_x_array = np.arange(self.spec_map.shape[-1]) #self.dat['wls']
        
    def scan_specific_setup(self):
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'Wavelength', units='nm')        
