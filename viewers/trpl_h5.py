from ScopeFoundry.data_browser import DataBrowser, HyperSpectralBaseView
import numpy as np
import h5py
from qtpy import QtWidgets
import pyqtgraph as pg
from scipy.optimize import least_squares
from ScopeFoundry.logged_quantity import LQCollection



class TRPLH5View(HyperSpectralBaseView):

    name = 'trpl_scan_h5'
    
    def is_file_supported(self, fname):
        for name in ["_trpl_2d_scan.h5", "_trpl_scan.h5", "Picoharp_MCL_2DSlowScan.h5"]:
            if name in fname: return True
        return False

    def load_data(self, fname):
        self.file = h5py.File(fname)
        
        load_success = False
        for measure_path in ['measurement/trpl_scan/', 'measurement/trpl_2d_scan/', 
                             'measurement/Picoharp_MCL_2DSlowScan/', 'measurement/APD_MCL_2DSlowScan/']:
            if  measure_path in self.file:
                self.H = self.file[measure_path]
                load_success = True
        if not load_success:
            print(self.H.items())
            raise ValueError("Measurement group not found in h5 file")
        
        self.S = self.file['hardware/picoharp/settings'].attrs
        
        try:
            cr0 = self.S['count_rate0']
            rep_period_s = 1.0/cr0
            time_bin_resolution = self.S['Resolution']*1e-12
            self.num_hist_chans = int(np.ceil(rep_period_s/time_bin_resolution))
        except:
            self.num_hist_chans = self.time_trace_map.shape[-1]
        
        
        t_slice = np.s_[0:self.num_hist_chans]
        
        time_array = np.array(self.H['time_array'])[t_slice]
        time_trace_map = np.array(self.H['time_trace_map'])[0,:,:,t_slice]
        integrated_count_map = time_trace_map.sum(axis=-1)

        # set defaults
        self.hyperspec_data = time_trace_map
        self.display_image = integrated_count_map
        self.spec_x_array = time_array
        
        print('load_data',self.hyperspec_data.shape)

        self.file.close()
        
    def post_load(self):
        self.recalc_taue_map()
        self.roll_offset.change_min_max(0,self.spec_x_array.shape[0])
        self.hyperspec_data = np.roll(self.hyperspec_data, self.settings['roll_offset'], -1)
        self.on_change_roll_max_to()
            
         
    def scan_specific_setup(self):
        # set spectral plot to be semilog-y
        self.spec_plot.setLogMode(False, True)
        self.spec_plot.setLabel('left', 'Intensity', units='counts')
        self.spec_plot.setLabel('bottom', 'time', units='ns')
        
        self.fit_on = self.settings.New('fit_on', str, initial = 'rect_roi',
                                          choices = ('None', 'rect_roi', 'circ'))
        self.fit_on.add_listener(self.on_change_fit_on)
        self.fit_line = self.spec_plot.plot(y=[0,2,1,3,2],pen='g')

        self.fit_option = self.settings.New('fit_option',str,initial='biexponential',
                                             choices = ('poly_fit','tau_x_calc', 'biexponential'))
        self.fit_option.add_listener(self.on_change_fit_option)
        
        self.fit_map_pushButton = QtWidgets.QPushButton(text = 'fit_map')
        self.settings_widgets.append(self.fit_map_pushButton)
        self.fit_map_pushButton.clicked.connect(lambda x:self.fit_map(update_display_image=True, fit_option=None))  
        
        self.roll_offset = self.settings.New('roll_offset', int, initial=0)
        self.roll_offset.add_listener(self.on_change_roll_offset)
        
        self.use_roll_max_to = self.settings.New('use_roll_max_to', bool, initial = True)
        self.roll_max_to = self.settings.New('roll_max_to', initial = 1)
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
        self.image_dock.raiseDock()  
        
        
    def on_change_roll_max_to(self):
        '''
        Note: might call a funciton which reloads the data
        '''        
        if self.use_roll_max_to.val:
            target_x = self.roll_max_to.val
            x,y = self.get_xy(np.s_[:,:],apply_use_x_slice=False)
            delta_index = np.argmin((x-target_x)**2) -  y.argmax()
            new_roll_offset = (self.roll_offset.val + delta_index) % x.shape[0]
            if new_roll_offset != self.roll_offset.val:
                self.roll_offset.update_value(new_roll_offset)

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
        if self.fit_on.val == 'None':
            alpha = 0
            self.x_slice_InfLineLabel.setText('x_slice')
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
        
    def fit_xy(self):
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
        fit_func_dict[fit_option](x,y)
        
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
        label_text = '\n'.join(['tau_x_calc', 'tau={:3.3f}'.format(tau)])
        self.x_slice_InfLineLabel.setText(label_text)
        self.fit_line.setData(x,y)
        return x,y

    def tau_x_calc_map(self, time_array, time_trace_map):
        return tau_x_calc_map(time_array - time_array.min(), time_trace_map)
        
    def poly_fit_xy(self,x,y,deg=1):       
        coefs = poly_fit(x=x, y=y)
        t = x - x.min()
        fit = np.exp( np.poly1d(coefs)(t) )    
        label_text = '\n'.join(['poly_fit','A={:3.3f}'.format(coefs[1]), 
                                'tau={:3.3f}'.format(-1/coefs[0])])
        self.x_slice_InfLineLabel.setText(label_text)
        self.fit_line.setData(x,fit)
        
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
        A0,tau0,A1,tau1 = order_bi_exp_components(A0, tau0, A1, tau1)
        tau_m = (A0*tau0 + A1*tau1) / (A0 + A1) 
        line0 = 'A_0 ={0:1.0f}, tau_0 ={1:1.2f}ns\n'.format(*bi_res.x[0:2].tolist())
        line1 = 'A_1 ={0:1.0f}, tau_1 ={1:1.2f}ns\n'.format(*bi_res.x[2:].tolist())
        line2 = 'tau_m ={0:1.2f}ns'.format(tau_m)        
        label_text = line0 + line1 + line2
        self.x_slice_InfLineLabel.setText(label_text)
        
        fit = biexponential(bi_res.x, t)
        self.fit_line.setData(x,fit)
        self.current_bi_exp_fit_res = bi_res.x

        
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
        A0,tau0,A1,tau1 = order_bi_exp_components(A0, tau0, A1, tau1)
        for key,image in zip(['A0_map','tau0_map','A1_map','tau1_map'],[A0,tau0,A1,tau1]):
            self.add_display_image(key, image)        
        taum = (A0*tau0 + A1*tau1) / (A0+A1)
        return(taum)

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
def order_bi_exp_components(A0,tau0,A1,tau1):
    '''
    ensures that tau0 > tau1, also swaps values in A1 and A0 if necessary.
    '''
    A0 = np.atleast_1d(A0)
    tau0 = np.atleast_1d(tau0)
    A1 = np.atleast_1d(A1)
    tau1 = np.atleast_1d(tau1) 
    mask = tau0 > tau1
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
    return new_A0,new_tau0,A1,tau1 #Note, generally A1,tau1 also modified.

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
    app.load_view(TRPLNPZView(app))
    
    sys.exit(app.exec_())