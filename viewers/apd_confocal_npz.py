from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
import pyqtgraph as pg
import numpy as np

class ApdConfocalNPZView(DataBrowserView):

    name = 'apd_confocal_npz'
    
    def setup(self):
        
        self.ui = self.imview = pg.ImageView()
        
        #self.graph_layout = pg.GraphicsLayoutWidget()
        #self.graph_layout.addPlot()
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = np.load(fname)
            self.imview.setImage(self.dat['count_rate_map'][::-1,:].T)
        except Exception as err:
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return "apd_confocal.npz" in fname

        
if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(ApdConfocalNPZView(app))
    
    sys.exit(app.exec_())