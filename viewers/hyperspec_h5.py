#from ScopeFoundry.data_browser import HyperSpectralBaseView
from FoundryDataBrowser.viewers.hyperspec_base_view import HyperSpectralBaseView
from FoundryDataBrowser.viewers.plot_n_fit import PeakUtilsFitter

import numpy as np
import h5py
import pyqtgraph as pg


class HyperSpecH5View(HyperSpectralBaseView):

    name = 'hyperspec_h5'

    supported_measurements = ['m4_hyperspectral_2d_scan',
                              'andor_hyperspec_scan',
                              'hyperspectral_2d_scan',
                              'fiber_winspec_scan',
                              'hyperspec_picam_mcl',
                              'asi_hyperspec_scan',
                              'asi_OO_hyperspec_scan',
                              'oo_asi_hyperspec_scan',
                              'andor_asi_hyperspec_scan',]

    def scan_specific_setup(self):
        pass

    def setup(self):
        self.settings.New('sample', dtype=str, initial='')
        HyperSpectralBaseView.setup(self)
        self.plot_n_fit.add_fitter(PeakUtilsFitter())

    def is_file_supported(self, fname):
        return np.any([(meas_name in fname)
                       for meas_name in self.supported_measurements])


    def reset(self):
        HyperSpectralBaseView.reset(self)
        if hasattr(self, 'dat'):
            self.dat.close()
            del self.dat

    def load_data(self, fname):
        print(self.name, 'loading', fname)
        self.dat = h5py.File(fname)
        for meas_name in self.supported_measurements:
            if meas_name in self.dat['measurement']:
                self.M = self.dat['measurement'][meas_name]

        self.spec_map = None
        for map_name in ['hyperspectral_map', 'spec_map']:
            if map_name in self.M:
                self.spec_map = np.array(self.M[map_name])
                if 'h_span' in self.M['settings'].attrs:
                    h_span = float(self.M['settings'].attrs['h_span'])
                    units = self.M['settings/units'].attrs['h0']
                    self.set_scalebar_params(h_span, units)
                    
                if len(self.spec_map.shape) == 4:
                    self.spec_map = self.spec_map[0, :, :, :]
                if 'dark_indices' in list(self.M.keys()):
                    self.spec_map = np.delete(self.spec_map,
                                              self.M['dark_indices'],
                                              -1)
        if self.spec_map is None:
            self.spec_map = np.zeros((10,10,10))
            raise ValueError("Specmap not found")
        self.hyperspec_data = self.spec_map
        self.display_image = self.hyperspec_data.sum(axis=-1)
        self.spec_x_array = np.arange(self.hyperspec_data.shape[-1])

        for x_axis_name in ['wavelength', 'wls', 'wave_numbers',
                            'raman_shifts']:
            if x_axis_name in self.M:
                x_array = np.array(self.M[x_axis_name])
                if 'dark_indices' in list(self.M.keys()):
                    x_array = np.delete(x_array,
                                        np.array(self.M['dark_indices']),
                                        0)
                self.add_spec_x_array(x_axis_name, x_array)
                self.x_axis.update_value(x_axis_name)
                
        sample = self.dat['app/settings'].attrs['sample']
        self.settings.sample.update_value(sample)



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
