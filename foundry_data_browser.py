from __future__ import absolute_import, print_function
from ScopeFoundry.data_browser import DataBrowser
import logging

logging.basicConfig(level=logging.DEBUG)#, filename='example.log', stream=sys.stdout)
logging.getLogger('traitlets').setLevel(logging.WARN)
logging.getLogger('ipykernel.inprocess').setLevel(logging.WARN)
logger = logging.getLogger('FoundryDataBrowser')

import sys

app = DataBrowser(sys.argv)


# views are loaded in order of more generic to more specific.
## ie the last loaded views are checked first for compatibility

from viewers.h5_tree import H5TreeView
app.load_view(H5TreeView(app))

try:
    from viewers.images import ScipyImreadView
    app.load_view(ScipyImreadView(app))
except ImportError:
    logger.warning("missing scipy")
    
from viewers.apd_confocal_npz import ApdConfocalNPZView, ApdConfocal3dNPZView
app.load_view(ApdConfocalNPZView(app))
app.load_view(ApdConfocal3dNPZView(app))

from viewers.picoharp_npz import PicoHarpNPZView
app.load_view(PicoHarpNPZView(app))

from viewers.hyperspec_npz import HyperSpecNPZView
app.load_view(HyperSpecNPZView(app))

from viewers.hyperspec_npz import HyperSpecSpecMedianNPZView
app.load_view(HyperSpecSpecMedianNPZView(app))


from viewers.trpl_npz import TRPLNPZView
app.load_view(TRPLNPZView(app))

from viewers.picoharp_mcl_2dslowscan import Picoharp_MCL_2DSlowScan_View
app.load_view(Picoharp_MCL_2DSlowScan_View(app))

from viewers.APD_MCL_2DSlowScanView import APD_MCL_2DSlowScanView
app.load_view(APD_MCL_2DSlowScanView(app))

from viewers.WinSpecMCL2DSlowScanView import WinSpecMCL2DSlowScanView
app.load_view(WinSpecMCL2DSlowScanView(app))

from viewers.WinSpecRemoteReadoutView import WinSpecRemoteReadoutView
app.load_view(WinSpecRemoteReadoutView(app))

from viewers.power_scan_h5 import PowerScanH5View
app.load_view(PowerScanH5View(app))


sys.exit(app.exec_())