from ScopeFoundry.data_browser import DataBrowser
from FoundryDataBrowser.viewers.hyperspec_base_view import HyperSpectralBaseView
import numpy as np
import h5py
from qtpy import QtWidgets
from ScopeFoundry.logged_quantity import LQCollection
import time

from FoundryDataBrowser.viewers.plot_n_fit import MonoExponentialFitter, BiExponentialFitter, SemiLogYPolyFitter, TauXFitter

class TRPLH5View(HyperSpectralBaseView):

    name = 'trpl_scan_h5'
    
    def is_file_supported(self, fname):
        for name in ["_trpl_2d_scan.h5", "_trpl_scan.h5", "Picoharp_MCL_2DSlowScan.h5"]:
            if name in fname: 
                self.fname = fname
                return True
        return False

    def load_data(self, fname):
        if hasattr(self, 'h5_file'):
            try:
                self.h5_file.close()
            except Exception as err:
                print("Could not close old h5 file:", err)
            del self.h5_file
        
        self.h5_file = h5py.File(fname)        
        load_success = False
        for measure_path in ['measurement/trpl_scan/', 'measurement/trpl_2d_scan/', 
                             'measurement/Picoharp_MCL_2DSlowScan/']:
            if  measure_path in self.h5_file:
                self.H = self.h5_file[measure_path]
                load_success = True
                
        if not load_success:
            raise ValueError(self.name, "Measurement group not found in h5 file", fname)
        
        
        # Note: The behavior of this viewer depends somewhat on the counting device:
        # see also set_hyperspec_data
        if 'counting_device' in self.H['settings'].attrs.keys():
            self.counting_device = self.H['settings'].attrs['counting_device']
        else:
            self.counting_device = 'picoharp'
        self.S = self.h5_file['hardware/{}/settings'.format(self.counting_device)].attrs
        
        time_array = self.H['time_array'][:] * 1e-3
        self.time_trace_map = self.H['time_trace_map'][:]
                
        # set defaults
        self.spec_x_array = time_array
        self.set_hyperspec_data()
        integrated_count_map = self.hyperspec_data.sum(axis=-1)
        self.display_image = integrated_count_map
        
                
        print(self.name, 'load_data of shape', self.hyperspec_data.shape)
        
        if 'dark_histogram' in self.H:
                self.dark_histogram = self.H['dark_histogram'][:]
                if np.ndim(self.dark_histogram)==1:
                    self.dark_histogram = np.expand_dims(self.dark_histogram, 0)
                self.bg_subtract.add_choices('dark_histogram')
        
        if 'h_span' in self.H['settings'].attrs:
            h_span = float(self.H['settings'].attrs['h_span'])
            units = self.H['settings/units'].attrs['h0']
            self.set_scalebar_params(h_span, units)

        self.h5_file.close()
        self.roll_offset.change_min_max(0, self.spec_x_array.shape[0])
        
        
    def set_hyperspec_data(self):
        # this function sets the hyperspec data based on self.time_trace_map
        # and (`chan`, `frame) setting. The shape of time_trace_map depends on counting device: 
        # 1. 4D picoharp data:  (Nframe, Ny, Nx, Ntime_bins)
        # 2. 5D hydraharp data: (Nframe, Ny, Nx, Nchan, Ntime_bins)
    
        if hasattr(self, 'time_trace_map'):
            shape = self.time_trace_map.shape
            n_frame = shape[0]
            self.settings.frame.change_min_max(0, n_frame-1)
            frame = self.settings['frame']
            if np.ndim(self.time_trace_map) == 5:
                n_chan = shape[-2]
                self.settings.chan.change_min_max(0, n_chan-1)
                hyperspec_data = self.time_trace_map[frame,:,:,self.settings['chan'],:]
            if np.ndim(self.time_trace_map) == 4:
                self.settings['chan'] = 0
                self.settings.chan.change_min_max(0, 0)
                hyperspec_data = self.time_trace_map[frame,:]
            
            roll_offset = self.roll_offset.val        
            if  roll_offset == 0:
                self.hyperspec_data = hyperspec_data
            else:   
                self.hyperspec_data = np.roll(hyperspec_data, self.settings['roll_offset'], -1)
                if hasattr(self, 'dark_histogram'):
                    self.dark_histogram = np.roll(self.dark_histogram, self.settings['roll_offset'], -1)        
        
        
    def add_descriptor_suffixes(self, key):
        #key += '_chan{}'.format(str(self.settings['chan']))
        return HyperSpectralBaseView.add_descriptor_suffixes(self, key)

                
    def get_bg(self):
        if self.bg_subtract.val == 'dark_histogram':
            bg = self.dark_histogram[self.settings['chan'],self.x_slicer.slice].mean()
            if not self.x_slicer.activated.val:
                self.x_slicer.activated.update_value(True)
                #self.x_slicer.set_label(title='dark_histogram bg', text=str(bg))
            return bg
        else:
            return HyperSpectralBaseView.get_bg(self)
               
            
    def scan_specific_setup(self):
        self.spec_plot.setLogMode(False, True)
        self.spec_plot.setLabel('left', 'Intensity')
        self.spec_plot.setLabel('bottom', 'time')
        
        S = self.settings        
        
        self.time_unit = self.settings.New('time_unit', str, initial = 'ns')
                
        self.settings.New('chan', dtype=int, initial=0, vmin=0)
        self.settings.chan.add_listener(self.set_hyperspec_data)
        
        self.settings.New('frame', dtype=int, initial=0, vmin=0)
        self.settings.frame.add_listener(self.set_hyperspec_data)
        
        self.roll_offset = self.settings.New('roll_offset', int, initial=0, unit='idx')
        self.roll_offset.add_listener(self.on_change_roll_offset)
        
        self.use_roll_x_target = self.settings.New('use_roll_max_to', bool, initial=False)
        self.roll_x_target = self.settings.New('roll_x_target', initial=1, unit='[x]')
        self.use_roll_x_target.add_listener(self.on_change_roll_x_target)
        self.roll_x_target.add_listener(self.on_change_roll_x_target)
                        

        self.export_settings = ES = LQCollection()
        ES.New('include_fit_results', bool, initial=True)
        ES.New('plot_title', str, initial='')
        ES.New('auto_y_lim', bool, initial = True)
        ES.New('y_lim_min', initial = -1)
        ES.New('y_lim_max', initial = -1)
        ES.New('auto_x_lim', bool, initial = True)
        ES.New('x_lim_min', initial = -1)
        ES.New('x_lim_max', initial = -1)
        export_ui = ES.New_UI()
        self.export_dock.addWidget( export_ui )
        
        self.export_plot_as_jpeg_pushButton = QtWidgets.QPushButton('export plot as jpeg')
        self.export_plot_as_jpeg_pushButton.clicked.connect(self.export_plot_as_jpeg)                  
        self.export_dock.addWidget( self.export_plot_as_jpeg_pushButton )       
        
        self.export_plot_as_xlsx_pushButton = QtWidgets.QPushButton('export plot as xlsx')
        self.export_plot_as_xlsx_pushButton.clicked.connect(self.export_plot_as_xlsx)    
        self.export_dock.addWidget( self.export_plot_as_xlsx_pushButton )     

        self.plot_n_fit.add_fitter(SemiLogYPolyFitter())            
        self.plot_n_fit.add_fitter(MonoExponentialFitter())
        self.plot_n_fit.add_fitter(BiExponentialFitter())
        self.plot_n_fit.add_fitter(TauXFitter())
                
        
    def on_change_roll_x_target(self):
        '''
        Note: might call a function which reloads the data
        '''        
        if self.use_roll_x_target.val:
            target_x = self.roll_x_target.val
            arr = self.time_trace_map
            y = arr.mean( tuple(range(arr.ndim-1)) )
            x = self.spec_x_array
            delta_index = np.argmin((x-target_x)**2) -  y.argmax()
            new_roll_offset = delta_index % x.shape[0]
            if new_roll_offset != self.roll_offset.val:
                self.roll_offset.update_value(new_roll_offset)
    
    def on_change_roll_offset(self):
        self.set_hyperspec_data()
        self.update_display()                                   
    
    def export_maps_as_jpegs(self):
        for name,image in self.display_images.items():
            if 'median' in name:
                cmap = 'rainbow'
            elif 'tau' in name:
                cmap = 'viridis'
            else:
                cmap = 'gist_heat'            
            self.export_image_as_jpeg(name, image, cmap)
    
    def save_fit_res_table(self, h5_file):
        res_table = self.plot_n_fit.get_result_table()
        h5_group = h5_file.create_group('fit_res_table')
        for (name, number, unit) in res_table:
            h5_group.attrs[name] = number
            h5_group.attrs[name + '_unit'] = unit
                    
    
    def gather_plot_data_for_export(self):        
        export_dict = {}
        if self.settings['show_circ_line']:
            x,y = self.get_xy(ji_slice=self.rect_roi_slice, apply_use_x_slice = False)
            x_shift = x[y.argmax()]
            export_dict.update({'point data':(x-x_shift,y)})
        if self.settings['show_rect_line']:
            x,y = self.get_xy(ji_slice=self.rect_roi_slice, apply_use_x_slice = False)
            x_shift = x[y.argmax()]                
            export_dict.update({'rectangle data':(x-x_shift,y)})
        P = self.plot_n_fit
        export_dict.update({'fit':(P.x_fit_data-x_shift, P.fit)})
        return export_dict


    def export_plot_as_xlsx(self):
        
        fname = self.databrowser.settings['data_filename']
        xlsx_fname = fname.replace( '.h5','_{:0.0f}.xlsx'.format(time.time()) )

        import xlsxwriter
        workbook = xlsxwriter.Workbook(xlsx_fname)

        worksheet = workbook.add_worksheet('data')
        for i,(label,(x,y)) in enumerate(self.gather_plot_data_for_export().items()):
            worksheet.write(0, i*2, label)
            for ii_, X in enumerate((x,y)):
                worksheet.write(1, i*2+ii_, ['time', 'counts'][ii_])
                worksheet.write_column(row=2, col=i*2+ii_, data = X)
                      
        if self.export_settings['include_fit_results']:
            worksheet = workbook.add_worksheet('fit_results')

            for i,row_ in enumerate(self.plot_n_fit.get_result_table()):
                worksheet.write_row(i,0,row_)
                
        workbook.close()
        self.databrowser.ui.statusbar.showMessage('exported data to ' + xlsx_fname)



    def export_plot_as_jpeg(self):
        print('export_plot_as_jpeg()')
        import matplotlib.pylab as plt
        ES = self.export_settings
                
        P = self.plot_n_fit
                
        L =  self.x_slicer.settings['stop'] - self.x_slicer.settings['start']
        
        plt.figure()
        ax = plt.subplot(111)
        
        y_lim = [None, None]
        x_lim = [None, None]
        
        for label,(x,y) in self.gather_plot_data_for_export().items():
            ax.semilogy(x,y, label = label)            
            if len(y) == L:
                y_lim = [0.9*y[-1], 1.05*y[0]]
                x_lim = [0.99*x[0], x[-1]*1.1]
            
                
        # Apply limits
        if ES['auto_y_lim']:
            ax.set_ylim(y_lim)
        else:
            ax.set_ylim(ES['y_lim_min'], ES['y_lim_max'])

        if ES['auto_x_lim']:
            ax.set_xlim(x_lim)
        else:
            ax.set_xlim(ES['x_lim_min'], ES['x_lim_max'])      
        
        plt.legend(loc=1)
        
        
        # Put the fit results somewhere
        if ES['include_fit_results']:
            tab = plt.table(cellText=P.get_result_table(),
                            colWidths=[0.15,0.1,0.04],
                            loc='lower left',
                            colLoc=['right','right','left'],
                            )
            tab.auto_set_font_size(True)
            for cell in tab.get_celld().values():
                cell.set_linewidth(0)
                
        if ES['plot_title'] != '':
            plt.title(ES['plot_title'])
        
        plt.xlabel('time ({})'.format(self.settings['time_unit']))
        plt.ylabel('intensity (a.u.)')
        plt.tight_layout()
        fname = self.databrowser.settings['data_filename']
        fig_name = fname.replace( '.h5','_{:0.0f}.jpg'.format(time.time()) )
        plt.savefig(fig_name, dpi=300)
        plt.close()    
        self.databrowser.ui.statusbar.showMessage('exported new data to ' + fig_name)

    
        
        





"""class TRPL3dNPZView(HyperSpectralBaseView):

    name = 'trpl_3d_npz'
    
    def setup(self):
        HyperSpectralBaseView.setup(self)
        TRPLNPZView.scan_specific_setup(self)
        
        self.settings.New('plane', dtype=str, initial='xy', choices=('xy', 'yz', 'xz'))
        self.settings.New('index', dtype=int)
        self.settings.New('auto_level', dtype=bool, initial=True)
        for name in ['plane', 'index', 'auto_level']:
            self.settings.get_lq(name).add_listener(self.update_display)
        
        #self.ui = QtWidgets.QWidget()
        #self.ui.setLayout(QtWidgets.QVBoxLayout())
        self.dockarea.addDock(name='Image', widget=self.settings.New_UI())
        self.info_label = QtWidgets.QLabel()
        self.dockarea.addDock(name='info', widget=self.info_label)
        #self.imview = pg.ImageView()
        #self.ui.layout().addWidget(self.imview, stretch=1)
        
        #self.graph_layout = pg.GraphicsLayoutWidget()
        #self.graph_layout.addPlot()
        
    def on_change_data_filename(self, fname):
        
        try:
            TRPLNPZView.load_data(self, fname)
            self.update_display()
        except Exception as err:
            self.imview.setImage(np.zeros((10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return "trpl_scan3d.npz" in fname
    
    def update_display(self):
        
        ii = self.settings['index']
        plane = self.settings['plane']
        
        if plane == 'xy':        
            arr_slice = np.s_[ii,:,:]
            index_max = self.dat['integrated_count_map'].shape[0]-1
        elif plane == 'yz':
            arr_slice = np.s_[:,:,ii]
            index_max = self.dat['integrated_count_map'].shape[2]-1
        elif plane == 'xz':
            arr_slice = np.s_[:,ii,:]
            index_max = self.dat['integrated_count_map'].shape[1]-1 

        self.settings.index.change_min_max(0, index_max)
        
        self.hyperspec_data = self.time_trace_map[:,:,:,0:self.num_hist_chans][arr_slice]+1
        self.display_image = self.integrated_count_map[arr_slice]
        
        
        #self.imview.setImage(self.dat['integrated_count_map'][arr_slice], autoLevels=self.settings['auto_level'], )

        other_ax = dict(xy='z', yz='x', xz='y' )[plane]

        self.info_label.setText("{} plane {}={} um (index={})".format(
            plane, other_ax, self.dat[other_ax+'_array'][ii], ii))
        
        HyperSpectralBaseView.update_display(self)"""

        

if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(TRPLH5View(app))
    
    sys.exit(app.exec_())