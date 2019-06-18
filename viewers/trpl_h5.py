from ScopeFoundry.data_browser import DataBrowser, HyperSpectralBaseView
import numpy as np
import h5py
from qtpy import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.optimize import least_squares
from ScopeFoundry.logged_quantity import LQCollection
from ScopeFoundry import h5_io
import time


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
            self.h5_file.close()
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
        
        
        if 'counting_device' in self.H['settings'].attrs.keys():
            self.counting_device = self.H['settings'].attrs['counting_device']
        else:
            self.counting_device = 'picoharp'
        self.S = self.h5_file['hardware/{}/settings'.format(self.counting_device)].attrs
        
        time_array = self.H['time_array'][:] * 1e-3
        self.time_trace_map = self.H['time_trace_map'][0]
                
        # set defaults
        self.set_hyperspec_data()
        self.spec_x_array = time_array
        
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
        
        
    def set_hyperspec_data(self):
        if np.ndim(self.time_trace_map) == 4:
            self.settings.chan.change_min_max(0,self.time_trace_map.shape[-2]-1)
            self.hyperspec_data = self.time_trace_map[:,:,self.settings['chan'],:]
        if np.ndim(self.time_trace_map) == 3:
            self.settings.chan.change_min_max(0,0)
            self.hyperspec_data = self.time_trace_map
            
        self.hyperspec_data = self.hyperspec_data
        integrated_count_map = self.hyperspec_data.sum(axis=-1)
        self.display_image = integrated_count_map
                
    def get_bg(self):
        if self.bg_subtract.val == 'dark_histogram':
            bg = self.dark_histogram[self.settings['chan'],self.x_slicer.slice].mean()
            if not self.x_slicer.activated.val:
                self.x_slicer.activated.update_value(True)
                #self.x_slicer.set_label(title='dark_histogram bg', text=str(bg))
            return bg
        else:
            return HyperSpectralBaseView.get_bg(self)
        
    def post_load(self):
        #self.recalc_taue_map()
        self.roll_offset.change_min_max(0,self.spec_x_array.shape[0])
        if  self.settings['roll_offset'] !=0:
            self.hyperspec_data = np.roll(self.hyperspec_data, self.settings['roll_offset'], -1)
            if hasattr(self, 'dark_histogram'):
                self.dark_histogram = np.roll(self.dark_histogram, self.settings['roll_offset'], -1)
            self.databrowser.ui.statusbar.showMessage('rolled data by: {} idxs'.format(self.settings['roll_offset']))
        self.on_change_roll_max_to()
            
            
    def scan_specific_setup(self):
        self.spec_plot.setLogMode(False, True)
        self.spec_plot.setLabel('left', 'Intensity')
        self.spec_plot.setLabel('bottom', 'time')
        self.time_unit = self.settings.New('time_unit', str, initial = 'ns')
        
        self.fit_on = self.settings.New('fit_on', str, initial = 'rect_roi',
                                          choices = ('None', 'rect_roi', 'circ'))
        self.fit_on.add_listener(self.on_change_fit_on)
        self.fit_line = self.spec_plot.plot(y=[0,2,1,3,2],pen='g')
        self.x_slicer.activated.add_listener(self.on_change_fit_on)
        self.x_slicer.region_changed_signal.connect(self.fit_xy)

        self.settings.New('chan', dtype=int, initial=0, vmin=0)
        self.settings.chan.add_listener(self.set_hyperspec_data)

        self.fit_option = self.settings.New('fit_option',str,initial='tau_x_calc',
                                             choices = ('poly_fit','tau_x_calc', 'biexponential'))
        self.fit_option.add_listener(self.on_change_fit_option)
        
        self.fit_map_pushButton = QtWidgets.QPushButton(text = 'fit_map')
        self.settings_widgets.append(self.fit_map_pushButton)
        self.fit_map_pushButton.clicked.connect(lambda x:self.fit_map(update_display_image=True, fit_option=None))  
        
        self.roll_offset = self.settings.New('roll_offset', int, initial=0, unit='idx')
        self.roll_offset.add_listener(self.on_change_roll_offset)
        
        self.use_roll_max_to = self.settings.New('use_roll_max_to', bool, initial = False)
        self.roll_max_to = self.settings.New('roll_max_to', initial = 1, unit='[x]')
        self.use_roll_max_to.add_listener(self.on_change_roll_max_to)
        self.roll_max_to.add_listener(self.on_change_roll_max_to)
        
        self.biexponential_settings = BS = LQCollection()

        BS.New('A0_initial', initial = 0.1)
        BS.New('tau0_initial', initial = 2.5)
        BS.New('A1_initial', initial = 10)
        BS.New('tau1_initial', initial = 6)
        
        BS.New('use_bounds', dtype = bool, initial = False)
        BS.New('A0_lower_bound', initial = 0)        
        BS.New('A0_upper_bound', initial = 1e10)        
        BS.New('tau0_lower_bound', initial = 0)
        BS.New('tau0_upper_bound', initial = 1e4)
        BS.New('A1_lower_bound', initial = 0)        
        BS.New('A1_upper_bound', initial = 1e10)        
        BS.New('tau1_lower_bound', initial = 0)
        BS.New('tau1_upper_bound', initial = 1e4)      
        
        for i in ['A0','tau0','A1','tau1']:
            getattr(BS, i+'_lower_bound').add_listener(self.set_biexponential_fit_bounds)
            getattr(BS, i+'_upper_bound').add_listener(self.set_biexponential_fit_bounds)
            getattr(BS, i+'_initial').add_listener(self.set_biexponential_fit_initials)
        BS.use_bounds.add_listener(self.set_biexponential_fit_bounds)
        self.set_biexponential_fit_bounds()
        self.set_biexponential_fit_initials()
                    
        self.biexponential_settings_ui = self.biexponential_settings.New_UI()
        self.dockarea.addDock(name='biexponential fitting settings', widget=self.biexponential_settings_ui,
                                   position='below', relativeTo=self.image_dock)
        
        self.set_last_biexponential_res_as_initials_pushButton = QtWidgets.QPushButton(text = 'use last result as initials')
        self.biexponential_settings_ui.layout().addWidget(self.set_last_biexponential_res_as_initials_pushButton)
        self.set_last_biexponential_res_as_initials_pushButton.clicked.connect(self.set_last_biexponential_res_as_initials)  
                
        self.export_settings = ES = LQCollection()
        ES.New('include_scale_bar', bool, initial = True)
        ES.New('scale_bar_width', initial=0.005, spinbox_decimals=3)
        ES.New('scale_bar_text', str, initial='auto')        
        ES.New('include_fit_results', bool, initial=True)
        ES.New('plot_title', str, initial='')
        ES.New('auto_y_lim', bool, initial = True)
        ES.New('y_lim_min', initial = -1)
        ES.New('y_lim_max', initial = -1)
        ES.New('auto_x_lim', bool, initial = True)
        ES.New('x_lim_min', initial = -1)
        ES.New('x_lim_max', initial = -1)
        
        self.export_settings_ui = self.export_settings.New_UI()
        self.dockarea.addDock(name='export settings', widget=self.export_settings_ui,
                                   position='below', relativeTo=self.settings_dock)

        self.export_maps_as_jpegs_pushButton = QtWidgets.QPushButton('export maps as jpegs')
        self.export_maps_as_jpegs_pushButton.clicked.connect(self.export_maps_as_jpegs)        
        self.export_settings_ui.layout().addWidget(self.export_maps_as_jpegs_pushButton)

        self.export_plot_as_jpeg_pushButton = QtWidgets.QPushButton('export plot as jpeg')
        self.export_plot_as_jpeg_pushButton.clicked.connect(self.export_plot_as_jpeg)          
        self.export_settings_ui.layout().addWidget(self.export_plot_as_jpeg_pushButton)
        
        self.export_plot_as_xlsx_pushButton = QtWidgets.QPushButton('export plot as xlsx')
        self.export_plot_as_xlsx_pushButton.clicked.connect(self.export_plot_as_xlsx)          
        self.export_settings_ui.layout().addWidget(self.export_plot_as_xlsx_pushButton)
                
        self.image_dock.raiseDock()
        self.settings_dock.raiseDock()
        
        
    def on_change_roll_max_to(self):
        '''
        Note: might call a function which reloads the data
        '''        
        if self.use_roll_max_to.val:
            target_x = self.roll_max_to.val
            x,y = self.get_xy(np.s_[:,:],apply_use_x_slice=False)
            delta_index = np.argmin((x-target_x)**2) -  y.argmax()
            new_roll_offset = (self.roll_offset.val + delta_index) % x.shape[0]
            if new_roll_offset != self.roll_offset.val:
                self.roll_offset.update_value(new_roll_offset)
        else:
            pass

    def on_change_roll_offset(self):
        '''
        reloads data, the actual rolling is done thereafter in self.post_load()!
        '''
        fname = self.databrowser.settings['data_filename']
        self.on_change_data_filename(fname)
        
    def recalc_taue_map(self):
        self.fit_map(fit_option='tau_x_calc')
        
    def on_change_fit_option(self):
        self.fit_xy()
                        
    def on_change_fit_on(self):
        if self.fit_on.val == 'None' or (not self.x_slicer.activated.val):
            alpha = 0
            self.x_slicer.set_label('')
        else:
            alpha = 1
        self.fit_line.setOpacity(alpha)
        self.fit_xy()

    def on_change_rect_roi(self):
        HyperSpectralBaseView.on_change_rect_roi(self)        
        self.fit_xy()

    def on_update_circ_roi(self):
        HyperSpectralBaseView.on_update_circ_roi(self)
        self.fit_xy()
    
    @QtCore.Slot()
    def fit_xy(self):
        if self.x_slicer.activated.value == False:
            return
        if self.fit_on.val == 'rect_roi':
            x,y = self.get_xy(self.rect_roi_slice, apply_use_x_slice=True)
        elif self.fit_on.val == 'circ':
            x,y = self.get_xy(self.circ_roi_slice, apply_use_x_slice=True)
        else:
            return
        fit_func_dict = {'poly_fit':  self.poly_fit_xy,
                         'tau_x_calc': self.tau_x_calc_xy,
                         'biexponential': self.fit_biexponential_xy}
        fit_option = self.settings['fit_option']
        self.xf,self.yf = fit_func_dict[fit_option](x,y)
        
        
        
    def fit_map(self, update_display_image = True, fit_option = None):
        time_array, time_trace_map = self.get_xhyperspec_data(apply_use_x_slice=True)
        fit_func_dict = {'poly_fit':  self.poly_fit_map,
                         'tau_x_calc': self.tau_x_calc_map,
                         'biexponential': self.fit_biexponential_map}
        if fit_option == None:
            fit_option = self.settings['fit_option']
        fit_map = fit_func_dict[fit_option](time_array, time_trace_map)
        new_map_name = fit_option+'_map'
        self.add_display_image(new_map_name, fit_map)
        if update_display_image:
            self.settings['display_image'] = new_map_name

    def tau_x_calc_xy(self,x,y):
        t = x.copy()
        t -= t.min()        
        tau = tau_x_calc(y, t)
        self.fit_line.setData(x,y)
        
        #gather result
        quantities = ['$\\tau_e$']
        numbers = '{0:1.1f}'.format(tau).split(" ")
        units = [self.settings['time_unit']]
        self.res_data_table = [[quantity, number, unit] for quantity, number, unit in zip(quantities,numbers,units)]
        self.x_slicer.set_label(_table2html(self.res_data_table, strip_latex=True), title='tau_x_calc')

        return x,y

    def tau_x_calc_map(self, time_array, time_trace_map):
        return tau_x_calc_map(time_array - time_array.min(), time_trace_map)
        
    def poly_fit_xy(self,x,y,deg=1):       
        coefs = poly_fit(x=x, y=y)
        t = x - x.min()
        fit = np.exp( np.poly1d(coefs)(t) )    
        self.fit_line.setData(x,fit)
        
        #gather result
        quantities = ['$A$','$\\tau$']
        numbers = '{0:1.1f} {1:1.1f}'.format(coefs[1],-1/coefs[0]).split(" ")
        units = ['-', self.settings['time_unit']]
        self.res_data_table = [[quantity, number, unit] for quantity, number, unit in zip(quantities,numbers,units)]
        self.x_slicer.set_label(_table2html(self.res_data_table, strip_latex=True), title='poly_fit')
        
        return x,fit
        
    def poly_fit_map(self, time_array, time_trace_map):
        coefs_map = poly_fit_map(time_array=time_array-time_array.min(), time_trace_map=time_trace_map)
        return -1/coefs_map[:,:,0]
    
    def set_biexponential_fit_bounds(self):
        if self.biexponential_settings['use_bounds']:
            lower_bounds = []
            for bound in ['A0','tau0','A1','tau1']:
                lower_bounds.append(self.biexponential_settings[bound+'_lower_bound'])
            upper_bounds = []
            for bound in ['A0','tau0','A1','tau1']:
                upper_bounds.append(self.biexponential_settings[bound+'_upper_bound'])
        else:
            lower_bounds = [-np.inf,-np.inf,-np.inf,-np.inf]
            upper_bounds = [ np.inf, np.inf, np.inf, np.inf]
        self.biexponential_fit_bounds = (lower_bounds,upper_bounds)
    
    def set_biexponential_fit_initials(self):
        biexponential_fit_initials = []
        for i in ['A0','tau0','A1','tau1']:
            biexponential_fit_initials.append(self.biexponential_settings[i+'_initial'])
        self.biexponential_fit_initials = biexponential_fit_initials
        
    def set_last_biexponential_res_as_initials(self):
        for i,val in zip(['A0','tau0','A1','tau1'],self.current_bi_exp_fit_res):
            self.biexponential_settings[i+'_initial'] = val
            
    def fit_biexponential_xy(self,x,y):
        bounds = self.biexponential_fit_bounds
        bi_initial = self.biexponential_fit_initials
        
        t = x - x.min()
                
        bi_res = least_squares(fun = biexponential_residuals,
                                 bounds  = bounds,
                                 x0 = bi_initial, 
                                 args = (t, y))

        A0,tau0,A1,tau1 = bi_res.x
        A0,tau0,A1,tau1 = sort_biexponential_components(A0, tau0, A1, tau1)
        
        A0_norm,A1_norm = A0/(A0 + A1),A1/(A0 + A1)
        tau_m = A0_norm*tau0 + A1_norm*tau1
        fit = biexponential(bi_res.x, t)
        self.fit_line.setData(x,fit)
        self.current_bi_exp_fit_res = bi_res.x


        quantities = ['$\\tau_0$','$\\tau_1$','$A_0$','$A_1$','$\\tau_m$']
        numbers = '{0:1.1f} {1:1.1f} {2:1.0f} {3:1.0f} {4:1.1f}'.format(tau0,tau1,A0_norm*100,A1_norm*100,tau_m).split(" ")
        time_unit = self.settings['time_unit']
        units = [time_unit, time_unit, '%', '%', time_unit]
        self.res_data_table = [[quantity, number, unit] for quantity, number, unit in zip(quantities,numbers,units)]
        self.x_slicer.set_label(_table2html(self.res_data_table, strip_latex=True),title='biexponential fit')
        
        return x,fit

    def fit_biexponential_map(self,time_array, time_trace_map):
        x,time_trace_map = self.get_xhyperspec_data(apply_use_x_slice=True)
        bounds = self.biexponential_fit_bounds
        bi_initial = self.biexponential_fit_initials
        t = x - x.min()
        bi_res_map = biexponential_map(t, time_trace_map, bi_initial, bounds, axis=-1)
        A0 = bi_res_map[:,:,0]
        tau0 = bi_res_map[:,:,1]
        A1 = bi_res_map[:,:,2]
        tau1 = bi_res_map[:,:,3]
        A0,tau0,A1,tau1 = sort_biexponential_components(A0, tau0, A1, tau1)
        for key,image in zip(['A0_map','tau0_map','A1_map','tau1_map'],[A0,tau0,A1,tau1]):
            self.add_display_image(key, image)        
        taum = (A0*tau0 + A1*tau1) / (A0+A1)
        return(taum)
    
    def view_specific_save_state_func(self, h5_file):
        h5_group_settings_group = h5_file.create_group('biexponential_settings')
        h5_io.h5_save_lqcoll_to_attrs(self.biexponential_settings, h5_group_settings_group)
        h5_group_settings_group = h5_file.create_group('export_settings')
        h5_io.h5_save_lqcoll_to_attrs(self.export_settings, h5_group_settings_group)
                                      
    def export_maps_as_jpegs(self):
        import matplotlib.pylab as plt
        for name,m in self.display_images.items():
            plt.figure(dpi=200)
            plt.title(name)
            if name in ['median']:
                cmap = 'rainbow'
            if name in ['tau_x_calc_map', 'tau0_map', 'tau1_map', 'biexponential_map', 'poly_fit_map']:
                cmap = 'viridis'
            else:
                cmap = 'gist_heat'
            ax = plt.subplot(111)
            plt.imshow(m, origin='lower', interpolation=None, cmap=cmap)
            ES = self.export_settings
            print(ES['scale_bar_text'])
            if ES['include_scale_bar']:
                add_scale_bar(ax, ES['scale_bar_width'], ES['scale_bar_text'])
            cb = plt.colorbar()
            plt.tight_layout()
            fig_name =  self.fname.replace('.h5','_{:0.0f}_{}.jpg'.format(time.time(),name) ) 
            plt.savefig(fig_name) 
            plt.close()
    
    
    def save_fit_res_table(self, h5_file):
        h5_group = h5_file.create_group('fit_res_table')
        for (name, number, unit) in self.res_data_table:
            h5_group.attrs[name] = number
            h5_group.attrs[name + '_unit'] = unit
            
        
    
    def gather_plot_data_for_export(self):        
        export_dict = {}
        #choose the data plot line that has a fit on it.
        if self.fit_on.val != 'None':
            if self.fit_on.val == 'rect_roi':                
                x,y = self.get_xy(ji_slice=self.rect_roi_slice, apply_use_x_slice = False)
            elif self.fit_on.val == 'circ':
                x,y = self.get_xy(ji_slice=self.circ_roi_slice, apply_use_x_slice=False)
            
            x_shift = x[y.argmax()]
            
            export_dict.update({'data':(x-x_shift,y)})
            export_dict.update({'fit':(self.xf-x_shift,self.yf)})
        #if there is no fit then choose according to show line settings
        elif self.fit_on.val == 'None':
            if self.settings['show_circ_line']:
                x,y = self.get_xy(ji_slice=self.rect_roi_slice, apply_use_x_slice = False)
                x_shift = x[y.argmax()]
                export_dict.update({'point data':(x-x_shift,y)})
            if self.settings['show_rect_line']:
                x,y = self.get_xy(ji_slice=self.rect_roi_slice, apply_use_x_slice = False)
                x_shift = x[y.argmax()]                
                export_dict.update({'rectangle data':(x-x_shift,y)})
        return export_dict


    def export_plot_as_xlsx(self):
        
        fname = self.databrowser.settings['data_filename']
        xlsx_fname = fname.replace( '.h5','_{:0.0f}_{}.xlsx'.format(time.time(), self.fit_on.val) )

        import xlsxwriter
        workbook = xlsxwriter.Workbook(xlsx_fname)

        worksheet = workbook.add_worksheet('data')
        for i,(label,(x,y)) in enumerate(self.gather_plot_data_for_export().items()):
            worksheet.write(0, i*2, label)
            for ii_, X in enumerate((x,y)):
                worksheet.write(1, i*2+ii_, ['time', 'counts'][ii_])
                worksheet.write_column(row=2, col=i*2+ii_, data = X)
                      
        if self.export_settings['include_fit_results'] and self.fit_on.val != 'None':
            worksheet = workbook.add_worksheet('fit_results')

            for i,row_ in enumerate(self.res_data_table):
                worksheet.write_row(i,0,row_)
                
        workbook.close()
        self.databrowser.ui.statusbar.showMessage('exported data to ' + xlsx_fname)


    def export_plot_as_jpeg(self):
        print('export_plot_as_jpeg()')
        import matplotlib.pylab as plt
        ES = self.export_settings
                
        plt.figure()
        ax = plt.subplot(111)
        y_lim = [0.99*self.yf[-1],  0.1] 

        for label,(x,y) in self.gather_plot_data_for_export().items():
            ax.semilogy(x,y, label = label)
            if y_lim[1] < 1.01*y.max():
                y_lim[1] = 1.01*y.max()
                
        # Apply limits
        if ES['auto_y_lim']:
            ax.set_ylim(ymin = y_lim[0])
        else:
            ax.set_ylim(ES['y_lim_min'],ES['y_lim_max'])

        if ES['auto_x_lim']:
            ax.set_xlim(xmax = 1.01*self.xf.max() )
        else:
            ax.set_xlim(ES['x_lim_min'],ES['x_lim_max'])      
        
        plt.legend(loc=1)
        
        # Put the fit results somewhere
        if ES['include_fit_results'] and self.fit_on.val != 'None':
            tab = plt.table(cellText=self.res_data_table,
                            colWidths=[0.15,0.1,0.04],
                            loc='lower left',
                            colLoc=['right','right','left'],
                            )
            tab.auto_set_font_size(True)
            for key, cell in tab.get_celld().items():
                cell.set_linewidth(0)
                
        if ES['plot_title'] != '':
            plt.title(ES['plot_title'])
        
        plt.xlabel('time ({})'.format(self.settings['time_unit']))
        plt.ylabel('intensity (a.u.)')
        plt.tight_layout()
        fname = self.databrowser.settings['data_filename']
        fig_name = fname.replace( '.h5','_{:0.0f}_{}.jpg'.format(time.time(), self.fit_on.val) )
        plt.savefig(fig_name, dpi=300)
        plt.close()    
        self.databrowser.ui.statusbar.showMessage('exported data to ' + fig_name)

        
        
def biexponential(params, t):
    '''
    params = [ A0, tau0, A1, tau1]    
    '''
    return params[0]*np.exp(-t/params[1]) + params[2]*np.exp(-t/params[3])
def biexponential_residuals(params, t, data):
    return biexponential(params,t) - data 
def fit_biexpontial(y, t,  bi_initial, bounds):
    bi_res = least_squares(fun = biexponential_residuals,
                                 bounds  = bounds,
                                 x0 = bi_initial, 
                                 args = (t, y))
    return bi_res.x    
def biexponential_map(t, time_trace_map, bi_initial, bounds, axis=-1):
    kwargs = dict(t=t, bi_initial=bi_initial, bounds=bounds)
    return np.apply_along_axis(fit_biexpontial, axis=axis, arr=np.squeeze(time_trace_map), **kwargs)
def sort_biexponential_components(A0,tau0,A1,tau1):
    '''
    ensures that tau0 < tau1, also swaps values in A1 and A0 if necessary.
    '''
    A0 = np.atleast_1d(A0)
    tau0 = np.atleast_1d(tau0)
    A1 = np.atleast_1d(A1)
    tau1 = np.atleast_1d(tau1) 
    mask = tau0 < tau1
    mask_ = np.invert(mask)
    new_tau0 = tau0.copy()
    new_tau0[mask_] = tau1[mask_]
    tau1[mask_] = tau0[mask_]
    new_A0 = A0.copy()
    new_A0[mask_] = A1[mask_]
    A1[mask_] = A0[mask_]
    try:
        new_A0 = np.asscalar(new_A0)
        new_tau0 = np.asscalar(new_tau0)
        A1 = np.asscalar(A1)
        tau1 = np.asscalar(tau1)
    except ValueError:
        pass
    return new_A0,new_tau0,A1,tau1 #Note, generally A1,tau1 were also modified.

def poly_fit(y,x,deg=1):
        mask = y > 0
        x = x[mask]
        y = y[mask]
        t = x.copy()
        t -= t.min()
        coefs = np.polyfit(t,np.log(y),deg)
        return coefs
def poly_fit_map(time_array, time_trace_map, axis=-1):
    kwargs = dict(x=time_array)
    return np.apply_along_axis(poly_fit, axis=axis, arr=np.squeeze(time_trace_map), **kwargs)

def tau_x_calc(time_trace, time_array, x=0.6321205588300001):
    t = time_trace
    return time_array[np.argmin(np.abs(np.cumsum(t)/np.sum(t)-x))]        
def tau_x_calc_map(time_array, time_trace_map, x=0.6321205588300001, axis=-1):
    kwargs = dict(time_array=time_array, x=x)
    return np.apply_along_axis(tau_x_calc, axis=axis, arr=np.squeeze(time_trace_map), **kwargs)

def add_scale_bar(ax, width=0.005, text=True, d=None, height=None, h_pos='left', v_pos='bottom',
                  color='w', edgecolor='k', lw=1, set_ticks_off=True, origin_lower=True, fontsize=13):
    from matplotlib.patches import Rectangle
    import matplotlib.pylab as plt
    imshow_ticks_off_kwargs = dict(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                         labelbottom=False, labeltop=False, labelleft=False, labelright=False)  
    """
    
        places a rectancle onto the axis *ax.
        d is the distance from the edge to rectangle.
    """
    
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    x1, y1 = ax.get_xlim()[1], ax.get_ylim()[1]
    
    Dx = x1 - x0
    if d == None:
        d = Dx / 18.
    if height == None:
        height = d * 0.8
    if width == None:
        width = 5 * d

    if h_pos == 'left':
        X = x0 + d
    else:
        X = x1 - d - width

    
    if origin_lower:
        if v_pos == 'bottom':
            Y = y0 + d
        else:
            Y = y1 - d - height
    else:
        if v_pos == 'bottom':
            Y = y0 - d - height
        else:
            Y = y1 + d

    xy = (X, Y)
    
    p = Rectangle(xy, width, height, color=color, ls='solid', lw=lw, ec=edgecolor)
    ax.add_patch(p)
    
    
    if text:
        if type(text) in [bool, None] or text == 'auto':
            text = str(int(width*1000)) + ' \u03BCm'
            print('caution: Assumes extent to be in mm, set text arg manually!')
        if v_pos == 'bottom':
            Y_text = Y+1.1*d
            va = 'bottom'
        else:
            Y_text = Y-0.1*d
            va = 'top'
        txt = plt.text(X+0.5*width,Y_text,text,
                 fontdict={'color':color, 'weight': 'heavy', 'size':fontsize,
                           #'backgroundcolor':edgecolor, 
                           'alpha':1, 'horizontalalignment':'center', 'verticalalignment':va}
                )
        import matplotlib.patheffects as PathEffects
        txt.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground=edgecolor)])    
    
    if set_ticks_off:
        ax.tick_params(**imshow_ticks_off_kwargs)
def _table2text(data_table, strip_latex = True):
    text = ''
    for line in data_table:
        text += (' '.join(line)+ '\n') 
    if strip_latex:
        text = text.replace('\\','').replace('$','').replace('_','')
    return text

def _table2html(data_table, strip_latex = True):
    text = '<table border="0">'
    for line in data_table:
        text += '<tr>'
        for element in line:
            text += '<td>{} </td>'.format(element)
        text += '</tr>'    
    text += '</table>'
    if strip_latex:
        text = text.replace('\\','').replace('$','').replace('_','')
    return text


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