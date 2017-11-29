from ScopeFoundry.data_browser import DataBrowser, HyperSpectralBaseView
import numpy as np
import h5py

class HyperSpecH5View(HyperSpectralBaseView):

    name = 'hyperspec_h5'
    
    def is_file_supported(self, fname):
        return "hyperspec_picam_mcl" in fname
    
    def load_data(self, fname):    
        self.dat = h5py.File(fname)
        self.M = self.dat['measurement/hyperspec_picam_mcl']
        print(self.M['spec_map'].shape)
        self.spec_map = self.M['spec_map'][0,:,:,:]
        
        self.integrated_count_map = self.spec_map.sum(axis=2)

        self.hyperspec_data = self.spec_map
        self.display_image = self.integrated_count_map
        self.spec_x_array = np.arange(self.spec_map.shape[-1])
        
    def scan_specific_setup(self):
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'Wavelength', units='nm')        


def spectral_median(spec,wls, count_min=200):
    int_spec = np.cumsum(spec)
    total_sum = int_spec[-1]
    if total_sum > count_min:
        pos = int_spec.searchsorted( 0.5*total_sum)
        wl = wls[pos]
    else:
        wl = np.NaN
    return wl
# 
# class HyperSpecSpecMedianH5View(HyperSpectralBaseView):
# 
#     name = 'hyperspec_spec_median_npz'
#     
#     def is_file_supported(self, fname):
#         return "_spec_scan.npz" in fname   
#     
#     
#     def load_data(self, fname):    
#         self.dat = np.load(fname)
#         
#         self.spec_map = self.dat['spec_map']
#         self.wls = self.dat['wls']
#         self.integrated_count_map = self.dat['integrated_count_map']
#         self.spec_median_map = np.apply_along_axis(spectral_median, 2,
#                                                    self.spec_map[:,:,:],
#                                                    self.wls, 0)
#         self.hyperspec_data = self.spec_map
#         self.display_image = self.spec_median_map
#         self.spec_x_array = self.wls
#         
#     def scan_specific_setup(self):
#         self.spec_plot.setLabel('left', 'Intensity', units='counts')
#         self.spec_plot.setLabel('bottom', 'Wavelength', units='nm')  
#         
# if __name__ == '__main__':
#     import sys
#     
#     app = DataBrowser(sys.argv)
#     app.load_view(HyperSpecH5View(app))
#     
#     sys.exit(app.exec_())