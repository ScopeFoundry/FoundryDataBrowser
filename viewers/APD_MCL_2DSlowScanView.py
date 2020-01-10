from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg
from qtpy import QtWidgets


class APD_MCL_2DSlowScanView(DataBrowserView):

    name = 'APD_MCL_2DSlowScan'
    
    def setup(self):
        
        self.ui = self.imview = pg.ImageView()
        self.imview.getView().invertY(False) # lower left origin
        
        #self.graph_layout = pg.GraphicsLayoutWidget()
        #self.graph_layout.addPlot()
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = h5py.File(fname, 'r')
            self.im_data = np.array(self.dat['measurement/APD_MCL_2DSlowScan/count_rate_map'][0,:,:].T) # grab first frame
            self.imview.setImage(self.im_data)
        except Exception as err:
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return ("APD_MCL_2DSlowScan.h5" in fname)


class APD_MCL_3DSlowScanView(DataBrowserView):

    name = 'APD_MCL_3DSlowScan'

    def setup(self):
        
        self.settings.New('plane', dtype=str, initial='xy', choices=('xy', 'yz', 'xz'))
        self.settings.New('index', dtype=int)
        self.settings.New('sub_frame', dtype=int)
        self.settings.New('auto_level', dtype=bool, initial=True)
        for name in ['plane', 'index', 'sub_frame', 'auto_level']:
            self.settings.get_lq(name).add_listener(self.update_display)
        
        self.ui = QtWidgets.QWidget()
        self.ui.setLayout(QtWidgets.QVBoxLayout())
        self.ui.layout().addWidget(self.settings.New_UI(), stretch=0)
        self.info_label = QtWidgets.QLabel()
        self.ui.layout().addWidget(self.info_label, stretch=0)
        self.imview = pg.ImageView()
        self.ui.layout().addWidget(self.imview, stretch=1)

    def is_file_supported(self, fname):
        return "APD_MCL_3DSlowScan.h5" in fname

    
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = h5py.File(fname, 'r')
            self.H = self.dat['measurement/APD_MCL_3DSlowScan/']
            h = self.h_settings = self.H['settings'].attrs
            self.im_data = np.array(self.H['count_rate_map'])
            self.settings.sub_frame.change_min_max(0, self.im_data.shape[1])

            self.stack_array = np.linspace(h['stack_min'], h['stack_max'], h['stack_num'])

            self.update_display()
        except Exception as err:
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)

    def update_display(self):
        
        if not hasattr(self, 'im_data'):
            return
        
        ii = self.settings['index']
        plane = self.settings['plane']
        sub_frame = self.settings['sub_frame']
        
        self.im_cube = self.im_data[:,sub_frame,:,:]
        
        H = self.H
        h = self.h_settings
        
        ax_lut = {
            h['stack_axis']: 0,
            h['v_axis']: 1,
            h['h_axis']: 2
        }
        
        arr_lut = {
            h['stack_axis']: self.stack_array,
            h['v_axis']: H['v_array'],
            h['h_axis']: H['h_array'],
        }
        
        other_ax = dict(xy='Z', yz='X', xz='Y' )[plane]

        arr_slice = [slice(None),slice(None),slice(None)]
        arr_slice[ax_lut[other_ax]]= ii
        a = arr_slice = tuple(arr_slice)
        
        index_max = self.im_cube.shape[ax_lut[other_ax]] -1

        self.settings.index.change_min_max(0, index_max)
        
        print(arr_slice)
        im = self.im_cube[arr_slice]
        print(self.im_cube.shape, im.shape)
        self.imview.setImage(im,
                             autoLevels=self.settings['auto_level'],axes=dict(x=1,y=0))

        
        other_ax_array = arr_lut[other_ax]

        self.info_label.setText("{} plane {}={:1.2f} um (index={})".format(
            plane, other_ax, other_ax_array[ii], ii))
    