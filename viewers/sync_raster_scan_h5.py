from ScopeFoundry.data_browser import DataBrowserView
import pyqtgraph as pg
import numpy as np
from qtpy import QtWidgets
import h5py
from collections import namedtuple, OrderedDict
import json

AvailChan = namedtuple('AvailChan', ['type_', 'index', 'phys_chan', 'chan_name', 'term'])


class SyncRasterScanH5(DataBrowserView):

    name = 'sync_raster_scan_h5'
    
    def setup(self):
        
        self.settings.New('frame', dtype=int)
        self.settings.New('sub_frame', dtype=int)
        self.settings.New('channel', dtype=str, choices=('ai0', 'ai1', 'ctr0', 'ctr1'))
        self.settings.New('auto_level', dtype=bool, initial=True)
        for name in ['frame', 'sub_frame','channel', 'auto_level']:
            self.settings.get_lq(name).add_listener(self.update_display)
        
        self.ui = QtWidgets.QWidget()
        self.ui.setLayout(QtWidgets.QVBoxLayout())
        self.ui.layout().addWidget(self.settings.New_UI(), stretch=0)
        self.info_label = QtWidgets.QLabel()
        self.ui.layout().addWidget(self.info_label, stretch=0)
        self.imview = pg.ImageView()
        self.ui.layout().addWidget(self.imview, stretch=1)
        
        #self.graph_layout = pg.GraphicsLayoutWidget()
        #self.graph_layout.addPlot()
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = h5py.File(fname)
            m_locs = ['measurement/sem_sync_raster_scan', 'measurement/sync_raster_scan']
            for node_name in m_locs:
                if node_name in self.dat:
                    M = self.measurement = self.dat[node_name]
            nframe, nsubframe, ny, nx, nadc_chan = M['adc_map'].shape
            self.settings.frame.change_min_max(0, nframe-1)
            self.settings.sub_frame.change_min_max(0, nsubframe-1)
            
            scanDAQ = self.dat['hardware/sync_raster_daq/settings'].attrs
            
            self.available_chan_dict = OrderedDict()
                
            for i, phys_chan in enumerate(json.loads(scanDAQ['adc_channels'])):
                self.available_chan_dict[phys_chan] = AvailChan(
                    # type, index, physical_chan, channel_name, terminal
                    'ai', i, phys_chan, json.loads(scanDAQ['adc_chan_names'])[i], phys_chan)
            for i, phys_chan in enumerate(json.loads(scanDAQ['ctr_channels'])):
                self.available_chan_dict[phys_chan] = AvailChan(
                    # type, index, physical_chan, channel_name, terminal
                    'ctr', i, phys_chan, json.loads(scanDAQ['ctr_chan_names'])[i], json.loads(scanDAQ['ctr_chan_terms'])[i])

            self.settings.channel.change_choice_list( [ (" ".join([chan.chan_name, chan.phys_chan]), key) for key, chan in self.available_chan_dict.items()] )
            
            self.update_display()
        except Exception as err:
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return ("sem_sync_raster_scan.h5" in fname) or ("sync_raster_scan.h5" in fname)
    
    def update_display(self):
        
        M = self.measurement
        
        ii = self.settings['frame']
        jj = self.settings['sub_frame']
        chan_name = self.settings['channel']

        chan = self.available_chan_dict[chan_name]
        
#         if   chan == 'ai0':
#             im = M['adc_map'][ii, jj, :,:, 0]
#         elif chan == 'ai1':
#             im = M['adc_map'][ii, jj, :,:, 1]
#         elif chan == 'ctr0':
#             im = M['ctr_map'][ii, jj, :,:, 0]
#         elif chan == 'ctr1':
#             im = M['ctr_map'][ii, jj, :,:, 1]

        if chan.type_ == 'ai':
            im = M['adc_map'][ii,jj,:,:,chan.index]
        if chan.type_ == 'ctr':
            im = M['ctr_map'][ii,jj,:,:,chan.index]


        self.imview.setImage(im.T[:,::-1], autoLevels=self.settings['auto_level'], )

        if self.settings['auto_level']:
            self.imview.setLevels(*np.percentile(im, (1,99) ))

        #self.info_label.setText("{} plane {}={} um (index={})".format(
        #    plane, other_ax, self.dat[other_ax+'_array'][ii], ii))
        