from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
import pyqtgraph as pg
import numpy as np
from scipy.misc import imread
import os

#scipy imread uses the Python Imaging Library (PIL) to read an image

class ScipyImreadView(DataBrowserView):

    name = 'scipy_imread_view'
    
    def setup(self):
        
        self.ui = self.imview = pg.ImageView()

        
    def on_change_data_filename(self, fname):
        
        try:
            self.data = imread(fname)
            self.imview.setImage(self.data.swapaxes(0,1))
        except Exception as err:
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        _, ext = os.path.splitext(fname)
        return ext.lower() in ['.png', '.tif', '.tiff', '.jpg']

