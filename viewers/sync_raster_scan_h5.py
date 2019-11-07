from ScopeFoundry.data_browser import DataBrowserView
import pyqtgraph as pg
import numpy as np
import h5py
from collections import namedtuple, OrderedDict
import json
from qtpy import QtWidgets
from .scalebars import SEMScaleBar


AvailChan = namedtuple('AvailChan', ['type_', 'index', 'phys_chan',
                                     'chan_name', 'term'])


class SyncRasterScanH5(DataBrowserView):

    name = 'sync_raster_scan_h5'

    supported_measurements = ['sem_sync_raster_scan',
                              'sync_raster_scan',
                              'hyperspec_cl']

    def setup(self):
        self.settings.New('frame', dtype=int)
        self.settings.New('sub_frame', dtype=int)
        self.settings.New('channel', dtype=str,
                          choices=('ai0', 'ai1', 'ctr0', 'ctr1'))
        self.settings.New('auto_level', dtype=bool, initial=True)
        self.settings.New('show_description', dtype=bool, initial=False)
        self.ui = QtWidgets.QWidget()
        self.ui.setLayout(QtWidgets.QVBoxLayout())
        self.ui.layout().addWidget(self.settings.New_UI(), stretch=0)
        self.info_label = QtWidgets.QLabel()
        self.ui.layout().addWidget(self.info_label, stretch=0)
        self.imitem = pg.ImageItem()
        self.imview = pg.ImageView(imageItem=self.imitem)

        self.ui.layout().addWidget(self.imview, stretch=1)

        for name in ['frame', 'sub_frame', 'channel', 'auto_level',
                     'show_description']:
            self.settings.get_lq(name).add_listener(self.update_display)

    def reset(self):
        if hasattr(self, 'dat'):
            self.dat.close()

        if hasattr(self, 'scalebar'):
            self.imview.getView().removeItem(self.scalebar)
            del self.scalebar

        if hasattr(self, 'desc_txt'):
            self.imview.getView().removeItem(self.desc_txt)
            del self.desc_txt

    def on_change_data_filename(self, fname):
        self.reset()

        try:
            self.dat = h5py.File(fname)
            for meas_name in self.supported_measurements:
                node_name = 'measurement/' + meas_name
                if node_name in self.dat:
                    self.M = self.measurement = self.dat[node_name]
            nframe, nsubframe, ny, self.nx, nadc_chan = self.M['adc_map'].shape
            self.settings.frame.change_min_max(0, nframe-1)
            self.settings.sub_frame.change_min_max(0, nsubframe-1)

            sem_remcon = self.dat['hardware/sem_remcon/settings'].attrs
            self.mag = sem_remcon['magnification'] / \
                (self.M['settings'].attrs['h_span']/20)

            scanDAQ = self.dat['hardware/sync_raster_daq/settings'].attrs

            self.available_chan_dict = OrderedDict()

            for i, phys_chan in enumerate(json.loads(scanDAQ['adc_channels'])):
                self.available_chan_dict[phys_chan] = AvailChan(
                    # type, index, physical_chan, channel_name, terminal
                    'ai', i, phys_chan,
                    json.loads(scanDAQ['adc_chan_names'])[i], phys_chan)
            for i, phys_chan in enumerate(json.loads(scanDAQ['ctr_channels'])):
                self.available_chan_dict[phys_chan] = AvailChan(
                    # type, index, physical_chan, channel_name, terminal
                    'ctr', i, phys_chan,
                    json.loads(scanDAQ['ctr_chan_names'])[i],
                    json.loads(scanDAQ['ctr_chan_terms'])[i])

            self.settings.channel.change_choice_list([
                (" ".join([chan.chan_name, chan.phys_chan]), key)
                for key, chan in self.available_chan_dict.items()])
            desc = self.M['settings'].attrs['description']
            self.desc_txt = pg.TextItem(text=desc)

            self.update_display()
        except Exception as err:
            self.imview.setImage(np.zeros((10, 10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s"
                                                      % (fname, err))
            raise(err)

    def is_file_supported(self, fname):
        for meas in self.supported_measurements:
            if meas + ".h5" in fname:
                return True
        return False

    def update_display(self):
        M = self.measurement

        ii = self.settings['frame']
        jj = self.settings['sub_frame']
        chan_name = self.settings['channel']

        chan = self.available_chan_dict[chan_name]

        nframe, nsubframe, ny, nx, nadc_chan = M['adc_map'].shape
#         if   chan == 'ai0':
#             im = M['adc_map'][ii, jj, :,:, 0]
#         elif chan == 'ai1':
#             im = M['adc_map'][ii, jj, :,:, 1]
#         elif chan == 'ctr0':
#             im = M['ctr_map'][ii, jj, :,:, 0]
#         elif chan == 'ctr1':
#             im = M['ctr_map'][ii, jj, :,:, 1]

        if chan.type_ == 'ai':
            im = M['adc_map'][ii, jj, :, :, chan.index]
        if chan.type_ == 'ctr':
            im = M['ctr_map'][ii, jj, :, :, chan.index]

        self.imitem.setImage(im.T[:, ::-1],
                             autoLevels=self.settings['auto_level'])
        if self.settings['auto_level']:
            self.imview.setLevels(*np.percentile(im, (1, 99)))

        if hasattr(self, 'desc_txt'):
            if self.settings['show_description']:
                self.imview.getView().addItem(self.desc_txt)
            else:
                self.imview.getView().removeItem(self.desc_txt)

        # calculate full frame size based on Polaroid 545 width (11.4cm)
        if hasattr(self, 'scalebar'):
            self.imview.getView().removeItem(self.scalebar)
        self.scalebar = SEMScaleBar(mag=self.mag, num_px=self.nx)
        self.scalebar.setParentItem(self.imview.getView())
        self.scalebar.anchor((1, 1), (1, 1), offset=(-20, -20))
        self.imview.autoRange()
        # self.info_label.setText("{} plane {}={} um (index={})".format(
        #    plane, other_ax, self.dat[other_ax+'_array'][ii], ii))
