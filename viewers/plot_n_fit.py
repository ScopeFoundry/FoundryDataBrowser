'''
Created on Aug 3, 2019

@author: Benedikt Ursprung
'''

import pyqtgraph as pg
import pyqtgraph.dockarea as dockarea

import numpy as np
from ScopeFoundry.logged_quantity import LQCollection
from qtpy import QtWidgets, QtCore, QtGui
from pySPM.utils import html_table


class PlotNFit(object):
    '''
    provides 2 pyqtgraph docks: 
        1. graph_dock: with plot
        2. settings_dock: with fit options
        
    add fitters of type <BaseFitter> (or more specific
    <LeastSquaresBaseFitter>):
    
    update_data(self, x, y, n_plot=0, is_fit_data=True) 
        to plot data. 
        if flag is_fit_data is False use:
            update_fit_data(self, x_fit_data, y_fit_data)
                this allows the data to differ.
    '''

    #fit_updated = QtCore.Signal()


    def __init__(self,
                 fitters=[],
                 Ndata_lines=1,
                 pens=['g', 'w', 'r', 'b', 'y', 'm', 'c']):
        '''
        *fitters*      list of <BaseFitter> or <LeastSquaresBaseFitter>
        *Ndata_lines*  number of data plots 
        *pens*         list of pens. p[0] will be used for fit line
                       and pens[1:] for data plot lines.   
        '''

        self.pens = pens

        self.x_fit_data = np.arange(4)
        self.y_fit_data = np.arange(4)

        # Settings
        self.settings = LQCollection()
        self.fit_options = self.settings.New(
            'fit_options', str, choices=['DisableFit'], initial='DisableFit')

        # layouts
        self.settings_ui = QtWidgets.QWidget()
        self.settings_layout = QtWidgets.QVBoxLayout()
        self.settings_layout.insertWidget(
            0, QtWidgets.QLabel('<h2>Plot&Fit</h2>'))
        self.settings_ui.setLayout(self.settings_layout)
        self.settings_dock = dockarea.Dock(
            name='Fit Settings', widget=self.settings_ui)
        self.settings_layout.addWidget(self.settings.New_UI())

        self.graph_layout = pg.GraphicsLayoutWidget()
        self.plot = self.graph_layout.addPlot()
        self.graph_dock = dockarea.Dock(
            name='Graph Plot', widget=self.graph_layout)

        self.data_lines = []
        for i in range(Ndata_lines):
            data_line = self.plot.plot(y=[0, 2, 1, 3, 2], pen=pens[i + 1])
            self.data_lines.append(data_line)
        self.fit_line = self.plot.plot(y=[0, 2, 1, 3, 2], pen=pens[0])
        self.vertical_lines = []

        #fitters
        self.fitter_uis = []
        for fitter in fitters:
            self.add_fitter(fitter)

        self.result_message = 'No fit results yet!'


        for lq in self.settings.as_list():
            lq.add_listener(self.on_change_fit_options)

        self.on_change_fit_options()
        
        self.add_button('refit', self.update_fit)
        self.add_button('clipboard plot', self.clipboard_plot)
        self.add_button('clipboard results', self.clipboard_result)
        VSpacerItem = QtWidgets.QSpacerItem(0, 0,
                                            QtWidgets.QSizePolicy.Minimum,
                                            QtWidgets.QSizePolicy.Expanding)
        self.settings_layout.addItem(VSpacerItem)

    def add_fitter(self, fitter):
        name = fitter.name
        print('PlotNFit is adding fitter', name)
        F = self.__dict__[name] = fitter
        self.fitter_uis.append([name, F.ui])
        self.settings_layout.addWidget(F.ui)
        F.ui.setVisible(False)
        self.fit_options.add_choices(name)

    def get_docks_as_dockarea(self):
        self.dockarea = dockarea.DockArea()
        self.add_docks_to_dockarea(self.dockarea)
        return self.dockarea

    def add_docks_to_dockarea(self, dockArea):
        dockArea.addDock(self.settings_dock)
        dockArea.addDock(
            self.graph_dock, position='right', relativeTo=self.settings_dock)
        self.settings_dock.setStretch(1, 1)

    def on_change_fit_options(self):
        choice = self.fit_options.val
        for name, ui in self.fitter_uis:
            ui.setVisible(choice == name)
        self.update_fit()

    def update_data(self, x, y, n_plot=0, is_fit_data=True):
        self.data_lines[n_plot].setData(x, y)
        self.x, self.y = x, y  #Note: this is the set data
        if is_fit_data:
            self.update_fit_data(x, y)

    def update_fit_data(self, x_fit_data, y_fit_data):
        self.x_fit_data = x_fit_data
        self.y_fit_data = y_fit_data
        self.update_fit()

    def update_fit(self):
        self.remove_lines()
        choice = self.fit_options.val
        enabled = choice != 'DisableFit'
        self.fit_line.setVisible(enabled)
        if enabled:
            x, y = self.x_fit_data, self.y_fit_data
            active_fitter = self.__dict__[choice]
            self.fit = active_fitter.fit_xy(x, y)
            self.fit_line.setData(x, self.fit)
            self.result_message = active_fitter.result_message
            self.set_vertical_lines(active_fitter)
            #self.fit_updated.emit()

    def remove_lines(self):
        for l in self.vertical_lines:
            self.plot.removeItem(l)
            l.deleteLater()
        self.vertical_lines = []
        
    def set_vertical_lines(self, active_fitter):
        self.remove_lines()
        pen = self.pens[0]

        for x in np.atleast_1d(active_fitter.highlight_x_vals):
            l = pg.InfiniteLine(
                pos=(x, 0),
                movable=False,
                angle=90,
                pen=pen,
                label='{value:0.2f}',
                labelOpts={
                    'color': pen,
                    'movable': True,
                    'fill': (200, 200, 200, 100)
                })
            self.plot.addItem(l)
            self.vertical_lines.append(l)

    def fit_hyperspec(self, x, _hyperspec, axis=-1):
        choice = self.fit_options.val
        if self.fit_options.val == 'DisableFit':
            print('Warning!', self.state_info)
        else:
            F = self.__dict__[choice]
            Res = F.fit_hyperspec(x, _hyperspec, axis=axis)
            Descriptions = F.hyperspec_descriptions()
            return [Descriptions, Res]

    def add_button(self, name, callback_func):
        PB = QtWidgets.QPushButton(name)
        self.settings_layout.addWidget(PB)
        PB.clicked.connect(callback_func)

    @property
    def state_info(self):
        choice = self.fit_options.val
        if choice == 'DisableFit':
            return 'Plot&Fit disabled'
        else:
            return self.__dict__[choice].state_info

    def get_result_table(self, decimals=3, include=None):
        choice = self.fit_options.val
        if choice == 'DisableFit':
            return 'Plot&Fit disabled'
        else:
            return self.__dict__[choice].get_result_table(decimals, include)
            
        
    def clipboard_plot(self):
        import pyqtgraph.exporters as exp
        exporter = exp.SVGExporter(self.plot)   
        exporter.parameters()['scaling stroke'] = False
        exporter.export(copy=True)
        
    def clipboard_result(self):

        html_string = self.result_message
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_string, 'lxml')
        table = soup.find_all('table')[0]
        text = ''
        for line in table.findAll('tr'):
            for l in line.findAll('td'):
                print(l.getText())
                text += l.getText()
    
        print(text)
        
        QtGui.QApplication.clipboard().setText(text)
        
        


class BaseFitter(object):
    '''    
        *fit_params* dict with all params and associated values: 
                     {
                     'ParamName0':(initial, lower_bound, upper_bound), 
                     ...'
                     } Note: may be empty    
                        
        *name*       any string 
        
        Implement `fit_xy(x,y)`
        
        see also <LeastSquaresBaseFitter>
        
        other useful function to override:
            process_results
            add_derived_result_quantities
    '''

    fit_params = {}
    name = 'xy_base_fitter'

    def __init__(self):

        self.fit_results = LQCollection()
        self.derived_results = LQCollection()
        self.bounds = LQCollection()
        self.initials = LQCollection()
        self.settings = LQCollection()

        if len(self.fit_params) == 0:
            print('Warning', self.name, 'has no fit_params')

        for name, (val, lower, upper) in self.fit_params.items():
            self.bounds.New(name + "_lower", initial=lower)
            self.bounds.New(name + "_upper", initial=upper)
            self.initials.New(name, initial=val)
            self.fit_results.New(name, initial=val)

        self.use_bounds = self.settings.New('use_bounds', bool, initial=False)

        self.add_derived_result_quantities()
        self.add_settings_quantities()

        layout = QtWidgets.QVBoxLayout()
        self.ui = ui = QtWidgets.QWidget()
        ui.setLayout(layout)
        for collection in ['settings', 'bounds', 'initials']:
            widget = self.__dict__[collection].New_UI()
            self.__dict__[collection + '_ui'] = widget
            if len(self.__dict__[collection].as_list()) != 0:
                layout.addWidget(QtWidgets.QLabel(f'<h3>{collection}</h3>'))
                layout.addWidget(widget)

        self.result_label = QtWidgets.QLabel()
        layout.addWidget(self.result_label)
        self.add_button('initials from results',
                        self.set_initials_from_results)

        self.bounds_ui.setEnabled(self.use_bounds.val)
        self.use_bounds.add_listener(lambda: self.bounds_ui.setEnabled(
            self.use_bounds.val))

        self.result_message = self.name + 'result_message message not updated'

        self.highlight_x_vals = []  # add x vals here.

    def fit_xy(self, x, y):
        ''' 
        has to return an array with the fit of len(y)
        recommended properties/functions to use:
            
            self.initials_array
            self.bounds_array
            
            self.update_fit_results(fit_results_array, additional_msg) 
                [recommended if the number of fit_params is fixed]
                otherwise pass a string to self.set_result_message(message)
        '''
        raise NotImplementedError()

    def update_fit_results(self, fit_results_array, additional_msg=''):
        '''
        helper function, which updates the results
        quantitiy collection and sets the result_message.
        the order of *results_array* is the 
        same as the order of self.fit_results, that is: 
        [fit_params[0], fit_params[1], ... ,] 
        
        Note: this function calls self.process_results
        before updating the results table.
                    
        alternatively pass a string to set_result_message(message)
        '''
        processed_fit_results = self.process_results(fit_results_array)
        for val, lq in zip(processed_fit_results, self.fit_results.as_list()):
            lq.update_value(val)
            
        res_table = self.get_result_table(decimals=3)
        header = ['param', 'value', 'unit']
        html_table = _table2html(res_table, header=header)
        if additional_msg != 0:
            msg = html_table + f'<p margin-top=5px>{additional_msg}</p>' 
            self.set_result_message(msg)
        else:
            self.set_result_message(html_table)

    def add_button(self, name, callback_func):
        PB = QtWidgets.QPushButton(name)
        self.ui.layout().addWidget(PB)
        PB.clicked.connect(callback_func)

    def add_derived_result_quantities(self):
        '''add results other than fit_params: self.derived_results.New('resX', ...), 
        use `process_results` to update the quantities here!
        '''
        pass

    def process_results(self, fit_results_array):
        '''
        calculate and set derived_results here, this function will be called
        after update_fit_results, Note: this function has to return the (processed) 
        fit results in the correct order.'''
        processed_fit_results_array = fit_results_array
        return processed_fit_results_array

    def add_settings_quantities(self):
        '''
        add results other than fit_params e.g: self.results.New('resX', ...), 
        set them in process_results
        '''
        pass

    @property
    def bounds_array(self):
        '''returns least_square style bounds array'''
        if self.settings['use_bounds']:
            f = filter(lambda lq: lq.name.endswith('lower'),
                       self.bounds.as_list())
            lower_bounds = [lq.val for lq in f]
            f = filter(lambda lq: lq.name.endswith('upper'),
                       self.bounds.as_list())
            upper_bounds = [lq.val for lq in f]
        else:
            N_bound_pairs = len(self.initials.as_list())
            lower_bounds = [-np.inf] * N_bound_pairs
            upper_bounds = [np.inf] * N_bound_pairs
        return np.array([lower_bounds, upper_bounds])

    @property
    def derived_results_array(self):
        return np.array([lq.val for lq in self.derived_results.as_list()])

    @property
    def fit_results_array(self):
        return np.array([lq.val for lq in self.fit_results.as_list()])

    @property
    def initials_array(self):
        return np.array([lq.val for lq in self.initials.as_list()])

    def hyperspec_descriptions(self):
        '''overwrite these if the results of fit_hyperspec is not
        fully described by fit_params!'''
        return [self.name + '_' + lq.name for lq in self.fit_results.as_list()]

    def fit_hyperspec(self, t, _hyperspec, axis=-1):
        '''
        intended for multidimensional arrays.
        Convention: this function should fit along axis and should 
                    return the params along that axis.
        '''
        raise NotImplementedError(self.name +
                                  '_fit_hyperspec() not implemented')

    def get_result_table(self, decimals=3, include=None):
        res_table = []
        for lq in list(self.fit_results.as_list()) + list(
                self.derived_results.as_list()):
            if include == None or lq.name in include:
                q = lq.name
                val = ('{:4.{prec}f}'.format(lq.val, prec=decimals))
                if lq.unit is not None:
                    unit = '{}'.format(lq.unit)
                else:
                    unit = ''
                res_table.append([q, val, unit])
        return res_table

    def set_result_message(self, message):
        self.result_message = message
        self.result_label.setText('<h3>results</h3>' + self.result_message)

    def set_initials_from_results(self):
        for k in self.initials.as_dict().keys():
            self.initials[k] = self.fit_results[k]

    @property
    def state_info(self):
        return self.name + self.state_description()

    def state_description(self):
        return ''


def _table2html(data_table,
                header=[],
                markup='border="0" alignment="center", cellspacing="2"'):
    if len(data_table) == 0: return ''
    text = f'<table {markup}>'
    if len(header) == len(data_table[0]):
        text += '<tr align="left">'
        for element in header:
            text += '<th>{} </th>'.format(element)
        text += '</tr>'
    for line in data_table:
        text += '<tr>'
        for element in line:
            text += '<td>{} </td>'.format(element)
        text += '</tr>'
    text += '</table>'
    return text


from scipy.optimize import least_squares


class LeastSquaresBaseFitter(BaseFitter):
    ''' 
    wrapper scipy.otimize.least_squares and 
        
        *fit_params* dict with all params and associated values: 
                     {
                     'ParamName0':(initial, lower_bound, upper_bound), 
                     ...'
                     } may be empty!
                        
        *name*       any string 
        
        Implement `func(params,x)`, 
            
        Note: x-x.min() is passed to least_squares(...)
    '''

    fit_params = {}
    name = 'least_square_base_fitter'

    def func(self, params, x):
        '''Override! This is the fit function!'''
        raise NotImplementedError(self.name + ' needs a fit function')

    def _residuals(self, params, x, data):
        return self.func(params, x) - data

    def fit_xy(self, x, y):

        t = x - x.min()

        res = least_squares(
            fun=self._residuals,
            bounds=self.bounds_array,
            x0=self.initials_array,
            args=(t, y))

        self.update_fit_results(res.x, res.message + f'<br>nfval:{res.nfev}')

        fit = self.func(res.x, t)
        return fit

    def fit_hyperspec(self, t, _hyperspec, axis=-1):
        def f(y, t):
            res = least_squares(
                fun=self._residuals,
                bounds=self.bounds_array,
                x0=self.initials_array,
                args=(t, y))
            return res.x

        fit = np.apply_along_axis(f, axis, _hyperspec, t=t - t.min())
        return np.rollaxis(fit, axis)


class TauXFitter(BaseFitter):

    fit_params = {
        'tau': (
            1.0,
            0.0,
            1e10,
        ),
    }

    name = 'tau_x'

    def fit_xy(self, x, y):
        t = x.copy()
        t -= t.min()
        tau = tau_x_calc(y, t)

        self.update_fit_results([tau])

        return y

    def fit_hyperspec(self, x, _hyperspec, axis=-1):
        return tau_x_calc_map(x, _hyperspec, axis=axis)


def tau_x_calc(time_trace, time_array, X=0.6321205588300001):
    t = time_trace
    return time_array[np.argmin(np.abs(np.cumsum(t) / np.sum(t) - X))]


def tau_x_calc_map(time_array, time_trace_map, X=0.6321205588300001, axis=-1):
    kwargs = dict(time_array=time_array, X=X)
    return np.apply_along_axis(
        tau_x_calc, axis=axis, arr=time_trace_map, **kwargs)


class PolyFitter(BaseFitter):

    name = 'poly'

    def add_settings_quantities(self):
        self.settings.New('deg', int, initial=1)

    def transform(self, x, y):
        return x, y

    def inverse_transform(self, x, y):
        return x, y

    def fit_xy(self, x, y):
        deg = self.settings['deg']

        x_, y_ = self.transform(x, y)

        coefs = np.polynomial.polynomial.polyfit(x_, y_, deg)
        fit_ = np.polynomial.polynomial.polyval(x_, coefs)
        x, fit = self.inverse_transform(x, fit_)

        res_table = []
        header = ['coef', 'value']
        for i, c in enumerate(coefs):
            res_table.append([f'a{i}', "{:3.3f}".format(c)])

        html_table = _table2html(res_table, header)
        self.set_result_message(html_table)

        return fit

    def fit_hyperspec(self, x, _hyperspec, axis=-1):

        x_, h_ = self.transform(x, _hyperspec)

        # polyfit takes 2D array and fits along dim 0.
        h_ = h_.swapaxes(axis, 0)
        shape = h_.shape[1:]
        h_ = h_.reshape((len(x), -1))

        deg = self.settings['deg']
        coefs = np.polynomial.polynomial.polyfit(x_, h_, deg)
        Res = coefs.reshape(-1, *shape).swapaxes(0, axis)
        return Res

    def hyperspec_descriptions(self):
        return [f'a{i}' for i in range(self.settings['deg'])]


class SemiLogYPolyFitter(PolyFitter):

    name = 'semilogy_poly'

    def transform(self, x, y):
        return x, np.log(y)

    def inverse_transform(self, x, y):
        return x, np.exp(y)


class MonoExponentialFitter(LeastSquaresBaseFitter):

    fit_params = {'A0': (1.0, 0.0, 1e10), 'tau0': (1.0, 0.0, 1e10)}
    name = 'mono_exponential'

    def func(self, params, x):
        return params[0] * np.exp(-x / params[1])


class BiExponentialFitter(LeastSquaresBaseFitter):

    fit_params = {
        'A0': (
            1.0,
            0.0,
            1e10,
        ),
        'tau0': (1.0, 0.0, 1e10),
        'A1': (1.0, 0.0, 1e10),
        'tau1': (9.9, 0.0, 1e10),
    }

    name = 'bi_exponetial'

    def func(self, params, x):
        return params[0] * np.exp(-x / params[1]) + params[2] * np.exp(
            -x / params[3])

    def add_derived_result_quantities(self):
        self.derived_results.New('tau_m', float, initial=10, unit='ns')
        self.derived_results.New('A0_pct', float, initial=10, unit='%')
        self.derived_results.New('A1_pct', float, initial=10, unit='%')

    def process_results(self, fit_results_array):
        A0, tau0, A1, tau1 = fit_results_array
        A0, tau0, A1, tau1 = sort_biexponential_components(A0, tau0, A1, tau1)

        A0_norm, A1_norm = A0 / (A0 + A1), A1 / (A0 + A1)
        tau_m = A0_norm * tau0 + A1_norm * tau1

        # update derived results
        D = self.derived_results
        D['tau_m'] = tau_m
        D['A0_pct'] = A0_norm * 100
        D['A1_pct'] = A1_norm * 100
        return (A0, tau0, A1, tau1)

    def fit_hyperspec(self, t, _hyperspec, axis=-1):
        A0, tau0, A1, tau1 = LeastSquaresBaseFitter.fit_hyperspec(
            self, t, _hyperspec, axis=axis)
        A0, tau0, A1, tau1 = sort_biexponential_components(A0, tau0, A1, tau1)

        A0_norm, A1_norm = A0 / (A0 + A1), A1 / (A0 + A1)
        tau_m = A0_norm * tau0 + A1_norm * tau1

        return np.array([A0_norm, tau0, A1_norm, tau1, tau_m])

    def hyperspec_descriptions(self):
        return LeastSquaresBaseFitter.hyperspec_descriptions(self) + ['taum']


def sort_biexponential_components(A0, tau0, A1, tau1):
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
    return new_A0, new_tau0, A1, tau1  #Note, generally A1,tau1 were also modified.


class PeakUtilsFitter(BaseFitter):

    name = 'peakUtils'

    def add_settings_quantities(self):
        self.settings.New('baseline_deg', int, initial=0, vmin=-1, vmax=100)
        self.settings.New('thres', float, initial=0.5, vmin=0, vmax=1)
        self.settings.New('unique_solution', bool, initial=False)
        self.settings.New('min_dist', int, initial=-1)
        self.settings.New('gaus_fit_refinement', bool, initial=True)
        self.settings.New('ignore_phony_refinements', bool, initial=True)

    def fit_xy(self, x, y):

        PS = self.settings
        import peakutils
        fit = base = 1.0 * peakutils.baseline(y, PS['baseline_deg'])

        if PS['min_dist'] < 0:
            min_dist = int(len(x) / 2)
        else:
            min_dist = PS['min_dist']
        peaks_ = peaks(y - base, x, PS['thres'], PS['unique_solution'],
                       min_dist, PS['gaus_fit_refinement'],
                       PS['ignore_phony_refinements'])

        peaks_ = np.atleast_1d(peaks_)
        self.highlight_x_vals = peaks_

        res_table = []
        header = ['peaks']
        for i, p in enumerate(peaks_):
            res_table.append(["{:3.3f}".format(p)])
        html_table = _table2html(res_table, header)
        self.set_result_message(html_table)

        return fit

    def fit_hyperspec(self, x, _hyperspec, axis=-1):
        PS = self.settings
        return peak_map(_hyperspec, x, axis, PS['thres'], int(len(x) / 2),
                        PS['gaus_fit_refinement'],
                        PS['ignore_phony_refinements'])

    def hyperspec_descriptions(self):
        return ['peak']

    def state_description(self):
        s = ''
        if self.settings['gaus_fit_refinement']:
            s += '_refined'
            if self.settings['ignore_phony_refinements']:
                s += '_ignored'
        return s


def peaks(spec,
          wls,
          thres=0.5,
          unique_solution=True,
          min_dist=-1,
          refinement=True,
          ignore_phony_refinements=True):
    import peakutils
    indexes = peakutils.indexes(spec, thres, min_dist=min_dist)
    if unique_solution:
        #we only want the highest amplitude peak here!
        indexes = [indexes[spec[indexes].argmax()]]

    if refinement:
        peaks_x = peakutils.interpolate(wls, spec, ind=indexes)
        if ignore_phony_refinements:
            for i, p in enumerate(peaks_x):
                if p < wls.min() or p > wls.max():
                    print(
                        'peakutils.interpolate() yielded result outside wls range, returning unrefined result'
                    )
                    peaks_x[i] = wls[indexes[i]]
    else:
        peaks_x = wls[indexes]

    if unique_solution:
        return peaks_x[0]
    else:
        return peaks_x


def peak_map(hyperspectral_data, wls, axis, thres, min_dist, refinement,
             ignore_phony_refinements):
    return np.apply_along_axis(
        peaks,
        axis,
        hyperspectral_data,
        wls=wls,
        thres=thres,
        unique_solution=True,
        min_dist=min_dist,
        refinement=refinement,
        ignore_phony_refinements=ignore_phony_refinements)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    W = PlotNFit(fitters=[
        BiExponentialFitter(),
        MonoExponentialFitter(),
        TauXFitter(),
        PolyFitter(),
        SemiLogYPolyFitter(),
        PeakUtilsFitter(),
    ])

    A = W.get_docks_as_dockarea()
    app.setActiveWindow(A)
    A.show()

    # Test latest fitter:
    x = np.arange(120) / 12

    y = np.exp(-x / 10.0) + 0.01 * np.random.rand(len(x))
    #y = x - 10 + 0.001 * np.random.rand(len(x))

    W.update_data(x, y)

    x, y, = x[5:110], y[5:110]
    W.update_fit_data(x, y)

    hyperspec = np.array([y, y, y, y, y, y]).reshape((3, 2, len(x)))

    print(W.fit_hyperspec(x, hyperspec, -1))

    import sys
    sys.exit(app.exec())