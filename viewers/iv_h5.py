from ScopeFoundry.data_browser import DataBrowserView
import pyqtgraph as pg
import pyqtgraph.dockarea as dockarea
import numpy as np
import h5py


class IVH5View(DataBrowserView):
    
    name = 'iv_h5_view'
    
    def is_file_supported(self, fname):
        return('iv.h5' in fname)
            
    def setup(self):
        self.ui = self.dockarea = dockarea.DockArea()

        self.iv_layout = pg.GraphicsLayoutWidget()
        self.iv_dock = self.dockarea.addDock(name='IV curve', widget=self.iv_layout)

        self.iv_plot = self.iv_layout.addPlot()
        self.iv_plot.setLabel('bottom', 'Voltage', units='V')
        self.iv_plot.setLabel('left', 'Current', units='Amps') 
        self.iv_plotdata = self.iv_plot.plot(y=[0,2,1,3,2])
        
    def on_change_data_filename(self, fname=None):
        if hasattr(self, 'h5_file'):
            try:
                self.h5_file.close()
            except:
                pass # Was already closed
            finally:
                del self.h5_file
        
        self.h5_file = h5py.File(fname)
        H = self.h5_file['measurement/photocurrrent_iv'] 
        I = H['I'][:]
        V = H['V'][:]
        self.h5_file.close()

        self.iv_plotdata.setData(V,I)       


from .trpl_h5 import TRPLH5View
class IVTRPLH5View(TRPLH5View):
    
    name = 'iv_trpl'
    
    def setup(self):
        TRPLH5View.setup(self)
        self.settings['use_roll_max_to'] = False
    
    def is_file_supported(self, fname):
        return 'iv_trpl.h5' in fname
    
    def load_data(self, fname):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
            del self.h5_file
    
        self.h5_file = h5py.File(fname)
        H = self.h5_file['measurement/iv_trpl']
        
        time_traces = H['time_traces'][:]
        #time_trace_map = np.zeros((2, *time_traces.shape))
        #time_trace_map[0,:] = time_traces
    
        time_trace_map = time_traces.reshape(2,-1,time_traces.shape[-1])
        time_trace_map[1] = time_trace_map[1,::-1,:]
        integrated_count_map = time_trace_map.sum(axis=-1)
        time_array = H['time_array'][:]*1e-3
        
        print(fname, time_trace_map.shape)
        print(integrated_count_map)
        
        self.hyperspec_data = time_trace_map
        self.display_image = integrated_count_map
        self.spec_x_array = time_array
        
        
        Vs = (H['V_sourced'][:]).reshape(integrated_count_map.shape)
        Vs[1] = Vs[1,::-1]
        self.add_display_image('V_sourced', Vs)

        
        Vm = (H['V'][:]).reshape(integrated_count_map.shape)
        Vm[1] = Vm[1,::-1]
        self.add_display_image('V_measured', Vm)
                
        Im = (H['I'][:]).reshape(integrated_count_map.shape)
        Im[1] = Im[1,::-1]
        self.add_display_image('I_measured', Im)
        
        self.h5_file.close()

            
class IVTRPLH5View_unfinished(IVH5View):
    
    def is_file_supported(self, fname):
        return 'iv_trpl.h5' in fname
    
    def setup(self):
        IVH5View.setup(self)
        
        self.trpl_layout = pg.GraphicsLayoutWidget()
        self.trpl_dock = self.dockarea.addDock(name='IV curve', widget=self.trpl_layout)

        self.trpl_plot = self.trpl_layout.addPlot()
        self.trpl_plot.setLabel('bottom', 'Voltage', units='V')
        self.trpl_plot.setLabel('left', 'Current', units='Amps') 
        self.trpl_plotdata = self.trpl_plot.plot(y=[0,2,1,3,2])    
        
    def on_change_data_filename(self, fname):
        IVH5View.on_change_data_filename(self, fname)
        
        