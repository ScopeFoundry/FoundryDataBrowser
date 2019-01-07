from ScopeFoundry.data_browser import DataBrowser, HyperSpectralBaseView
import numpy as np
import h5py
from qtpy import QtCore, QtWidgets
import pyqtgraph as pg

   

class HyperSpecH5View(HyperSpectralBaseView):

    name = 'hyperspec_h5'

    def scan_specific_setup(self):
        self.spec_plot.setLabel('left', 'Intensity', units='counts') 
        
        self.integrated_count_map = np.array([[1,2],[3,4]])
        self.median_map = np.array([[1,1],[2,2]])
        
        self.img_choices = ['integrated_count_map']
        self.settings.New('display_image', str, choices=self.img_choices, initial='integrated_count_map')    
        self.settings.display_image.add_listener(self.on_display_image_change)
        
        self.x_axis = self.settings.New('x_axis', str, choices=('',))
        self.x_axis.add_listener(self.on_x_axis_change)

        # add settings as a dock
        self.settings_ui = self.settings.New_UI()
        ds = self.dockarea.addDock(name='settings', widget=self.settings_ui,
                                   position='left', relativeTo=self.image_dock)
        ds.setStretch(1,2)
        
        # add correlation plot as a dock
        self.corr_layout = pg.GraphicsLayoutWidget()
        self.corr_plot = self.corr_layout.addPlot()
        self.corr_plot.setLabel('left', 'intensity', units='')  
        self.corr_plot.setLabel('bottom', 'median wls', units='nm') 
        
        self.corr_plotdata = self.corr_plot.plot() 
        self.dockarea.addDock(name='correlation', widget=self.corr_layout, 
                              position='below',  relativeTo = self.spec_dock)
        self.spec_dock.raiseDock()
        
        
    def is_file_supported(self, fname):
        return np.any( [(meas_name in fname)
                            for meas_name in ['m4_hyperspectral_2d_scan', 'andor_hyperspec_scan', 
                                              'hyperspectral_2d_scan', 'fiber_winspec_scan',
                                              'hyperspec_picam_mcl.h5','asi_hyperspec_scan']])

    def load_data(self, fname):        
        
        self.h5_file = h5py.File(fname)
        for meas_name in ['m4_hyperspectral_2d_scan', 'hyperspectral_2d_scan', 'andor_hyperspec_scan', 
                          'fiber_winspec_scan','asi_hyperspec_scan', 'hyperspec_picam_mcl']:
            if meas_name in self.h5_file['measurement']:
                self.M = self.h5_file['measurement'][meas_name]

        for map_name in ['hyperspectral_map', 'spec_map']:
            if map_name in self.M:
                m = np.array(self.M[map_name])
                if len(m.shape) == 4:
                    m = m[0,:,:,:]
                    
        self.hyperspec_data = m
        self.integrated_count_map = self.hyperspec_data.sum(axis=2)

        self.x_axis_choices = []
        for x_axis_choice in ['wavelength', 'wls', 'wave_numbers', 'raman_shifts']:
            try:
                setattr(self, x_axis_choice, np.array(self.M[x_axis_choice]) )
                self.x_axis_choices.append(x_axis_choice)
            except Exception as err:
                #print("failed to find {} array".format(x_axis_choice))
                pass
        if len(self.x_axis_choices) == 0:
            print("failed to find any x_axis choices, generated generic pixels array")
            self.pixels = np.arange(self.hyperspec_data.shape[-1])
            self.x_axis_choices.append('pixels')
            
        self.x_axis.change_choice_list(self.x_axis_choices)
        self.x_axis.update_value(self.x_axis_choices[-1])            


        #try to add median map
        if hasattr(self, 'wls'):
            self.median_map = spectral_median_map(map_=self.hyperspec_data, wls=self.wls)
            
        if hasattr(self, 'wavelengths'):
            self.median_map = spectral_median_map(map_=self.hyperspec_data, wls=self.wavelengths) 

        if hasattr(self, 'median_map'):
            self.settings.display_image.change_choice_list(self.img_choices + ['median_map'])
            self.corr_plotdata.setData(x=self.integrated_count_map.flatten(), y=self.median_map.flatten())
        

        self.on_display_image_change()
        self.on_x_axis_change()
        
        self.h5_file.close()


      

    def on_display_image_change(self):
        #print('on_display_image_change')
        if hasattr(self, 'display_image'):
            del self.display_image

        self.display_image = getattr(self, self.settings['display_image'])
#         if self.settings['display_image'] == self.img_choices[1]:
#             cm = matplotlib_colormap_to_pg_colormap('rainbow')
#         else:
#             cm = matplotlib_colormap_to_pg_colormap('viridis')
#         self.imview.setColorMap(cm)
        self.update_display()


    def on_x_axis_change(self):        
        x_axis = self.settings['x_axis']
        self.spec_x_array = getattr(self, x_axis)
        self.spec_plot.setLabel('bottom', x_axis)
        self.update_display()


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
    Converts a Matplotlib cmap to pyqtgraphs colormaps. No dependency on matplotlib.
    Parameters:
    *cmap*: Cmap object. Imported from matplotlib.cm.*
    *nTicks*: Number of ticks to create when dict of functions is used. Otherwise unused.
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

def spectral_median(spec, wls, count_min=200):
    int_spec = np.cumsum(spec)
    total_sum = int_spec[-1]
    if total_sum > count_min:
        pos = int_spec.searchsorted( 0.5*total_sum)
        wl = wls[pos]
    else:
        wl = 0
    return wl
def spectral_median_map(map_, wls):
    return np.apply_along_axis(spectral_median,-1, map_, wls=wls)

def norm(x):
    x_max = x.max()
    if x_max==0:
        return x*0.0
    else:
        return x*1.0/x_max
def norm_map(map_):
    return np.apply_along_axis(norm, -1, map_)
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