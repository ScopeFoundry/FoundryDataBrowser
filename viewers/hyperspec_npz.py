from ScopeFoundry.data_browser import DataBrowser, HyperSpectralBaseView
import numpy as np

class HyperSpecNPZView(HyperSpectralBaseView):

    name = 'hyperspec_npz'
    
    def is_file_supported(self, fname):
        return "_spec_scan.npz" in fname   
    
    
    def load_data(self, fname):    
        self.dat = np.load(fname)
        
        self.spec_map = self.dat['spec_map']
        self.integrated_count_map = self.dat['integrated_count_map']

        self.hyperspec_data = self.spec_map
        self.display_image = self.integrated_count_map
        self.spec_x_array = self.dat['wls']
        
    def scan_specific_setup(self):
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'Wavelength', units='nm')        
        
if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(HyperSpecNPZView(app))
    
    sys.exit(app.exec_())