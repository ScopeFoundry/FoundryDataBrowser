from ScopeFoundry.data_browser import HyperSpectralBaseView
import h5py
import numpy as np
from .scalebars import SEMScaleBar
import pyqtgraph as pg


class HyperSpecCLH5View(HyperSpectralBaseView):

    name = 'hyperspec_cl_h5'

    def is_file_supported(self, fname):
        return "_hyperspec_cl.h5" in fname

    def setup(self):
        self.settings.New('show_description', dtype=bool, initial=False)
        self.settings.show_description.add_listener(self.update_display)
        HyperSpectralBaseView.setup(self)

    def load_data(self, fname):
        self.dat = h5py.File(fname)
        self.M = self.dat['measurement/hyperspec_cl']
        self.spec_map = self.M['spec_map'][0, 0, :, :, :]
        (self.nx, ny, nw) = self.spec_map.shape
        remcon = self.dat['hardware/sem_remcon/settings'].attrs
        self.spec_x_array = np.array(self.M['wls'])
        self.mag = remcon['magnification'] / \
            (self.M['settings'].attrs['h_span']/20)
        self.hyperspec_data = self.spec_map
        self.display_image = self.spec_map.sum(axis=-1)
        desc = self.M['settings'].attrs['description']
        self.desc_txt = pg.TextItem(text=desc)

    def update_display(self):
        if self.display_image is not None and hasattr(self, 'dat'):
            # pyqtgraph axes are x,y, but data is stored in (frame, y,x, time),
            # so we need to transpose
            self.imview.getImageItem().setImage(self.display_image.T)

            if hasattr(self, 'scalebar'):
                self.imview.getView().removeItem(self.scalebar)

            self.scalebar = SEMScaleBar(mag=self.mag, num_px=self.nx)
            self.scalebar.setParentItem(self.imview.getView())
            self.scalebar.anchor((1, 1), (1, 1), offset=(-20, -20))

            if hasattr(self, 'desc_txt'):
                if self.settings['show_description']:
                    self.imview.getView().addItem(self.desc_txt)
                else:
                    self.imview.getView().removeItem(self.desc_txt)

            self.on_change_rect_roi()
            self.on_update_circ_roi()

    def reset(self):
        if hasattr(self, 'dat'):
            self.dat.close()

        if hasattr(self, 'scalebar'):
            self.imview.getView().removeItem(self.scalebar)
            del self.scalebar

        if hasattr(self, 'desc_txt'):
            self.imview.getView().removeItem(self.desc_txt)
            del self.desc_txt

        HyperSpectralBaseView.reset(self)

    def scan_specific_setup(self):
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'Wavelength', units='nm')
