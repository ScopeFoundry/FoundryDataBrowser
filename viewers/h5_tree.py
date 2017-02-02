from ScopeFoundry.data_browser import DataBrowserView
from qtpy import QtWidgets
import h5py


class H5TreeView(DataBrowserView):

    name = 'h5_tree'
    
    def is_file_supported(self, fname):
        return ('.h5' in fname)
    
    def setup(self):
        self.ui = QtWidgets.QTextEdit("file_info")
    
    def on_change_data_filename(self, fname=None):
        self.ui.setText("loading {}".format(fname))
        try:        
            self.f = h5py.File(fname, 'r')
            
            self.tree_str = "{}\n{}\n".format(fname, "="*len(fname)) 
            self.f.visititems(self._visitfunc)
            self.ui.setText(self.tree_str)
            
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)

    def _visitfunc(self, name, node):
        
        level = len(name.split('/'))
        indent = '    '*level
        localname = name.split('/')[-1]
    
        if isinstance(node, h5py.Group):
            self.tree_str += indent +"|> {}\n".format(localname)
        elif isinstance(node, h5py.Dataset):
            self.tree_str += indent +"|D {}: {} {}\n".format(localname, node.shape, node.dtype)
        for key, val in node.attrs.items():
            self.tree_str += indent+"    |- {} = {}\n".format(key, val)        