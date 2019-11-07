from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg
import pyqtgraph.dockarea as dockarea


class SpectrumH5View(DataBrowserView):

    name = 'spectrum_h5'

    def setup(self):
        self.ui = self.dockarea = dockarea.DockArea()
        self.settings.New('sample', dtype=str, initial='', ro=True)
        self.settings.New('spec_min', dtype=float, initial=0, ro=True)
        self.settings.New('spec_max', dtype=float, initial=1, ro=True)
        self.settings.spec_max.add_listener(self.update_display)
        self.settings.spec_min.add_listener(self.update_display)
        self.settings_ui = self.settings.New_UI()
        self.dockarea.addDock(name='settings', position='top',
                              widget=self.settings_ui)
        self.plot = pg.PlotWidget(title="Spectrum")
        self.dockarea.addDock(name='plot', widget=self.plot)
        self.plot_setup()

    def on_change_data_filename(self, fname):
        try:
            self.dat = h5py.File(fname, 'r')
            self.H = self.dat['measurement/oo_spec_live']
            self.settings.sample.update_value(
                str(self.dat['app']['settings'].attrs['sample']))
            self.spectrum = self.H['spectrum']
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage(
                "failed to load %s:\n%s" % (fname, err))
            raise(err)
        try:
            self.wavelengths = np.array(self.M['wavelengths'])
        except Exception as err:
            print("failed to find wls array", err)
            self.wavelengths = np.arange(np.size(self.spectrum))
            self.settings.spec_min.change_readonly(ro=False)
            self.settings.spec_max.change_readonly(ro=False)
        self.settings.spec_min.update_value(self.wavelengths.min())
        self.settings.spec_max.update_value(self.wavelengths.max())

        self.update_display()

    def is_file_supported(self, fname):
        return "oo_spec_live.h5" in fname

    def plot_setup(self):
        ''' create plots for channels and/or sum'''
        self.plot_line = self.plot.plot()
        self.plot.setLabel('left', 'Intensity', units='counts')
        self.plot.setLabel('bottom', 'Wavelength', units='nm')

    def update_display(self):
        spec_min = self.settings.spec_min.val
        spec_max = self.settings.spec_max.val
        spec_vals = np.size(self.spectrum)
        if spec_min < spec_max and spec_min != 0 and spec_max != spec_vals-1:
            self.wavelengths = np.linspace(spec_min, spec_max, num=spec_vals)

        self.plot_line.setData(self.wavelengths, self.spectrum)
