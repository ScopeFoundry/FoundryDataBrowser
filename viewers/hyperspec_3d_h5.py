from ScopeFoundry.data_browser import HyperSpectralBaseView
import numpy as np
import h5py
import pyqtgraph as pg
from .scalebars import ConfocalScaleBar
from matplotlib.cm import ScalarMappable
from pyqtgraph.opengl import GLViewWidget, GLAxisItem, GLGridItem, GLVolumeItem
from scipy.interpolate import interp1d
from qtpy.QtWidgets import QPushButton
import time


class HyperSpec3DH5View(HyperSpectralBaseView):

    name = 'hyperspec_3d_h5'

    supported_measurements = ['oo_asi_hyperspec_3d_scan',
                              'andor_asi_hyperspec_3d_scan',]

    def scan_specific_setup(self):
        pass

    def setup(self):
        self.settings.New('sample', dtype=str, initial='')
        self.settings.New('z_slice', dtype=float, choices=[0.0], initial=0.0)
        self.settings.New('show_3d', dtype=bool, initial=False)
        self.settings.New('vol_alpha', dtype=float, vmin=0.0, vmax=1.0,
                          initial=0.5)
        self.settings.New(
            'vol_colormap', dtype=str, initial='viridis',
            choices=['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                     'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])
        # self.settings.New('vol_percentile', dtype=int, vmin=0, vmax=49,
        #                   initial=5)
        self.settings.New('vol_percentile_min', dtype=int, vmin=0, vmax=100,
                          initial=5)
        self.settings.New('vol_percentile_max', dtype=int, vmin=0, vmax=100,
                          initial=95)
        self.settings.New('vol_transparent_percentile', dtype=int, vmin=0,
                          vmax=100, initial=5)
        self.settings.New('vol_transparent_min', dtype=bool, initial=False)
        self.settings.z_slice.updated_choice_index_value.connect(
            self.on_update_zslice_choice)
        # self.settings.vol_colormap.updated_value.connect(self.calculate_volume)
        # self.settings.vol_alpha.updated_value.connect(self.calculate_volume)
        HyperSpectralBaseView.setup(self)
        voldata = np.empty((1, 1, 1, 4), dtype=np.ubyte)
        voldata[0, 0, 0, :] = [255, 255, 255, 0]
        self.volitem = GLVolumeItem(data=voldata)
        self.glview = GLViewWidget()
        self.glaxis = GLAxisItem()
        self.glgrid = GLGridItem()
        self.glview.addItem(self.glgrid)
        self.glview.addItem(self.glaxis)
        self.glview.addItem(self.volitem)
        self.gldock = self.dockarea.addDock(name='3D', widget=self.glview,
                                            position='below',
                                            relativeTo=self.image_dock)

        self.calculate_3d_pushButton = QPushButton(text='calculate_3d')
        self.settings_ui.layout().addWidget(self.calculate_3d_pushButton)
        self.calculate_3d_pushButton.clicked.connect(self.calculate_volume)
        self.image_dock.raiseDock()

    def is_file_supported(self, fname):
        return np.any([(meas_name in fname)
                       for meas_name in self.supported_measurements])

    def reset(self):
        if hasattr(self, 'dat'):
            self.dat.close()
            del self.dat

        if hasattr(self, 'spec_map'):
            del self.spec_map

        if hasattr(self, 'scalebar'):
            self.imview.getView().removeItem(self.scalebar)
            del self.scalebar

        if hasattr(self, 'volume'):
            spoof_data = np.zeros((1, 1, 1, 4), dtype=np.ubyte)
            self.volitem.setData(spoof_data)
            del self.volume

        self.settings.show_3d.update_value(False)
        self.image_dock.raiseDock()

    def load_data(self, fname):
        self.dat = h5py.File(fname)
        for meas_name in self.supported_measurements:
            if meas_name in self.dat['measurement']:
                self.M = self.dat['measurement'][meas_name]

        for map_name in ['hyperspectral_map', 'spec_map']:
            if map_name in self.M:
                self.spec_map = np.array(self.M[map_name])
                self.h_span = self.M['settings'].attrs['h_span']
                self.x_array = np.array(self.M['h_array'])
                self.z_array = np.array(self.M['z_array'])
                units = self.M['settings/units'].attrs['h_span']
                if units == 'mm':
                    self.h_span = self.h_span*1e-3
                    self.z_span = self.z_array*1e-3
                    self.settings.z_slice.change_unit('mm')

                if 'dark_indices' in list(self.M.keys()):
                    print('dark indices found')
                    dark_indices = self.M['dark_indices']
                    if dark_indices.len() == 0:
                        self.spec_map = np.delete(
                            self.spec_map, list(dark_indices.shape), -1)
                    else:
                        self.spec_map = np.delete(
                            self.spec_map, np.array(dark_indices), -1)
                else:
                    print('no dark indices')

        self.hyperspec_data = self.spec_map[0, :, :, :]
        self.display_image = self.hyperspec_data.sum(axis=-1)
        self.settings.z_slice.change_choice_list(self.z_array.tolist())
        self.settings.z_slice.update_value(self.z_array[0])
        self.spec_x_array = np.arange(self.hyperspec_data.shape[-1])

        for x_axis_name in ['wavelength', 'wls', 'wave_numbers',
                            'raman_shifts']:
            if x_axis_name in self.M:
                x_array = np.array(self.M[x_axis_name])
                if 'dark_indices' in list(self.M.keys()):
                    dark_indices = self.M['dark_indices']
                    # The following is to read a dataset I initialized
                    # incorrectly for dark pixels. This can be replaced with
                    # the else statement entirely now that the measurement is
                    # fixed, but I still have a long measurement that will
                    # benefit from this.
                    if dark_indices.len() == 0:
                        x_array = np.delete(
                            x_array, list(dark_indices.shape), 0)
                    else:
                        x_array = np.delete(x_array, np.array(dark_indices), 0)
                self.add_spec_x_array(x_axis_name, x_array)
                self.x_axis.update_value(x_axis_name)

        sample = self.dat['app/settings'].attrs['sample']
        self.settings.sample.update_value(sample)
        self.calculate_volume()

    def on_update_zslice_choice(self, index):
        if hasattr(self, 'spec_map'):
            self.hyperspec_data = self.spec_map[index, :, :, :]
            self.display_images['default'] = self.hyperspec_data
            self.display_images['sum'] = self.hyperspec_data.sum(axis=-1)
            self.spec_x_arrays['default'] = self.spec_x_array
            self.spec_x_arrays['index'] = np.arange(
                self.hyperspec_data.shape[-1])
            self.recalc_bandpass_map()
            self.recalc_median_map()
            self.update_display()

    def calculate_volume(self):
        if not self.settings['show_3d']:
            print('calculate_volume called without show_3d')
            return

        print('calculating 3d volume')
        t0 = time.time()

        if hasattr(self, 'volume'):
            del self.volume

        if hasattr(self, 'mappable'):
            self.mappable.set_cmap(self.settings['vol_colormap'])
        else:
            self.mappable = ScalarMappable(cmap=self.settings['vol_colormap'])

        z_span = self.z_array[-1] - self.z_array[0]
        dx = self.x_array[1] - self.x_array[0]
        z_interp_array = np.linspace(np.amin(self.z_array),
                                     np.amax(self.z_array),
                                     num=z_span/dx)
        z_interp = None
        self.volume = None
        nz = len(z_interp_array)

        if self.settings['display_image'] == 'bandpass_map':
            print('bandpass_map selected')
            x, slice = self.get_xhyperspec_data(apply_use_x_slice=True)
            ind_min = np.nonzero(self.spec_x_array == x[0])[0][0]
            ind_max = np.nonzero(self.spec_x_array == x[-1])[0][0]
            data = np.zeros((len(self.z_array),) + slice.shape)
            data = self.spec_map[:, :, :, ind_min:ind_max]
            # for kk in range(len(self.z_array)):
            #     print(
            #         'grabbing bandpass layer %d of %d' % (kk, len(self.z_array)))
            #     self.settings.z_slice.update_value(self.z_array[kk])
            #     x, data[kk, :, :, :] = self.get_xhyperspec_data(
            #         apply_use_x_slice=True)
            z_interp = interp1d(self.z_array, data, axis=0)
        else:
            z_interp = interp1d(self.z_array, self.spec_map, axis=0)

        data = z_interp(z_interp_array)
        self.volume = np.zeros(data.shape[:-1] + (4,), dtype=np.ubyte)

        pmin = self.settings['vol_percentile_min']
        pmax = self.settings['vol_percentile_max']
        self.mappable.set_array(data.sum(axis=-1))
        vmin = np.percentile(data.sum(axis=-1), pmin)
        vmax = np.percentile(data.sum(axis=-1), pmax)
        tmin = np.percentile(
            data.sum(axis=-1),
            self.settings['vol_transparent_percentile'])
        self.mappable.set_clim(vmin=vmin, vmax=vmax)
        # self.mappable.autoscale()

        for kk in range(nz):
            print('calculating rgba vals for %d of %d layers' % (kk, nz))
            sum_data = data[kk, :, :, :].sum(axis=-1)
            # print(sum_data.shape, self.volume.shape)
            self.volume[kk, :, :, :] = self.mappable.to_rgba(
                sum_data,
                alpha=self.settings['vol_alpha'],
                bytes=True
            )
            if self.settings['vol_transparent_min']:
                self.volume[kk, :, :, 3][np.nonzero(sum_data <= tmin)] = 0

        print('3d volume calculation complete')
        t1 = time.time()
        print('time elapsed: %0.3f s' % (t1-t0))

        kwargs = {'x': len(self.x_array), 'y': len(self.x_array), 'z': nz}
        self.glaxis.setSize(**kwargs)
        self.glgrid.setSize(**kwargs)
        self.glgrid.setSpacing(x=1/dx*5, y=1/dx*5, z=1/dx*5)
        # print(self.mappable.get_cmap().name)
        # print(data.shape, self.volume.shape)

    def update_display(self):
        if hasattr(self, 'scalebar'):
            self.imview.getView().removeItem(self.scalebar)

        if self.display_image is not None:
            # pyqtgraph axes are x,y, but data is stored in (frame, y,x, time),
            # so we need to transpose
            self.imview.getImageItem().setImage(self.display_image.T)

            nn = self.display_image.shape

            if hasattr(self, 'h_span'):
                span = self.h_span
            else:
                span = -1
            self.scalebar = ConfocalScaleBar(span=span, num_px=nn[0])
            self.scalebar.setParentItem(self.imview.getView())
            self.scalebar.anchor((1, 1), (1, 1), offset=(-20, -20))

            if hasattr(self, 'volume') and self.settings['show_3d']:
                self.volitem.setData(np.swapaxes(self.volume, 0, 2))

            self.on_change_rect_roi()
            self.on_update_circ_roi()


def matplotlib_colormap_to_pg_colormap(colormap_name, n_ticks=16):
    '''
    ============= =========================================================
    **Arguments**
    colormap_name (string) name of a matplotlib colormap i.e. 'viridis'

    n_ticks       (int)  Number of ticks to create when dict of functions
                  is used. Otherwise unused.
    ============= =========================================================

    returns:        (pgColormap) pyqtgraph colormap
    primary Usage:  <pg.ImageView>.setColorMap(pgColormap)
    requires:       cmapToColormap by Sebastian Hoefer
                    https://github.com/pyqtgraph/pyqtgraph/issues/561
    '''
    from matplotlib import cm
    pos, rgba_colors = zip(*cmapToColormap(getattr(cm, colormap_name)), n_ticks)
    pgColormap = pg.ColorMap(pos, rgba_colors)
    return pgColormap


def cmapToColormap(cmap, nTicks=16):
    """
    Converts a Matplotlib cmap to pyqtgraphs colormaps. No dependency on
    matplotlib.

    Parameters:
        *cmap*: Cmap object. Imported from matplotlib.cm.*
        *nTicks*: Number of ticks to create when dict of functions is used.
        Otherwise unused.

    author: Sebastian Hoefer
    """
    import collections
    # Case #1: a dictionary with 'red'/'green'/'blue' values as list of ranges (e.g. 'jet')
    # The parameter 'cmap' is a 'matplotlib.colors.LinearSegmentedColormap' instance ...
    if hasattr(cmap, '_segmentdata'):
        colordata = getattr(cmap, '_segmentdata')
        if ('red' in colordata) and isinstance(colordata['red'], collections.Sequence):

            # collect the color ranges from all channels into one dict to get unique indices
            posDict = {}
            for idx, channel in enumerate(('red', 'green', 'blue')):
                for colorRange in colordata[channel]:
                    posDict.setdefault(colorRange[0], [-1, -1, -1])[idx] = colorRange[2]

            indexList = list(posDict.keys())
            indexList.sort()
            # interpolate missing values (== -1)
            for channel in range(3):  # R,G,B
                startIdx = indexList[0]
                emptyIdx = []
                for curIdx in indexList:
                    if posDict[curIdx][channel] == -1:
                        emptyIdx.append(curIdx)
                    elif curIdx != indexList[0]:
                        for eIdx in emptyIdx:
                            rPos = (eIdx - startIdx) / (curIdx - startIdx)
                            vStart = posDict[startIdx][channel]
                            vRange = (posDict[curIdx][channel] - posDict[startIdx][channel])
                            posDict[eIdx][channel] = rPos * vRange + vStart
                        startIdx = curIdx
                        del emptyIdx[:]
            for channel in range(3):  # R,G,B
                for curIdx in indexList:
                    posDict[curIdx][channel] *= 255

            rgb_list = [[i, posDict[i]] for i in indexList]

        # Case #2: a dictionary with 'red'/'green'/'blue' values as functions (e.g. 'gnuplot')
        elif ('red' in colordata) and isinstance(colordata['red'], collections.Callable):
            indices = np.linspace(0., 1., nTicks)
            luts = [np.clip(np.array(colordata[rgb](indices), dtype=np.float), 0, 1) * 255 \
                    for rgb in ('red', 'green', 'blue')]
            rgb_list = zip(indices, list(zip(*luts)))

    # If the parameter 'cmap' is a 'matplotlib.colors.ListedColormap' instance, with the attributes 'colors' and 'N'
    elif hasattr(cmap, 'colors') and hasattr(cmap, 'N'):
        colordata = getattr(cmap, 'colors')
        # Case #3: a list with RGB values (e.g. 'seismic')
        if len(colordata[0]) == 3:
            indices = np.linspace(0., 1., len(colordata))
            scaledRgbTuples = [(rgbTuple[0] * 255, rgbTuple[1] * 255, rgbTuple[2] * 255) for rgbTuple in colordata]
            rgb_list = zip(indices, scaledRgbTuples)

        # Case #4: a list of tuples with positions and RGB-values (e.g. 'terrain')
        # -> this section is probably not needed anymore!?
        elif len(colordata[0]) == 2:
            rgb_list = [(idx, (vals[0] * 255, vals[1] * 255, vals[2] * 255)) for idx, vals in colordata]

    # Case #X: unknown format or datatype was the wrong object type
    else:
        raise ValueError("[cmapToColormap] Unknown cmap format or not a cmap!")

    # Convert the RGB float values to RGBA integer values
    return list([(pos, (int(r), int(g), int(b), 255)) for pos, (r, g, b) in rgb_list])


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
