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

from FoundryDataBrowser.viewers.h5_tree import H5TreeView
app.load_view(H5TreeView(app))

from FoundryDataBrowser.viewers.gauss2d_fit_img import Gauss2DFitImgView, Gauss2DFitAPD_MCL_2dSlowScanView
app.load_view(Gauss2DFitImgView(app))
app.load_view(Gauss2DFitAPD_MCL_2dSlowScanView(app))


try:
    from viewers.images import ScipyImreadView
    app.load_view(ScipyImreadView(app))
except ImportError:
    logger.warning("missing scipy")
    
from FoundryDataBrowser.viewers.apd_confocal_npz import ApdConfocalNPZView, ApdConfocal3dNPZView
app.load_view(ApdConfocalNPZView(app))
app.load_view(ApdConfocal3dNPZView(app))

from FoundryDataBrowser.viewers.picoharp_npz import PicoHarpNPZView
app.load_view(PicoHarpNPZView(app))

from viewers.hyperspec_npz import HyperSpecNPZView
app.load_view(HyperSpecNPZView(app))

from FoundryDataBrowser.viewers.hyperspec_npz import HyperSpecSpecMedianNPZView
app.load_view(HyperSpecSpecMedianNPZView(app))

from FoundryDataBrowser.viewers.trpl_t_x_lifetime import TRPL_t_x_lifetime_NPZView, TRPL_t_x_lifetime_fiber_scan_View
app.load_view(TRPL_t_x_lifetime_NPZView(app))
app.load_view(TRPL_t_x_lifetime_fiber_scan_View(app))

from FoundryDataBrowser.viewers.trpl_npz import TRPLNPZView, TRPL3dNPZView
app.load_view(TRPLNPZView(app))
app.load_view(TRPL3dNPZView(app))

from FoundryDataBrowser.viewers.picoharp_mcl_2dslowscan import Picoharp_MCL_2DSlowScan_View, FiberPicoharpScanView
app.load_view(Picoharp_MCL_2DSlowScan_View(app))
app.load_view(FiberPicoharpScanView(app))

from FoundryDataBrowser.viewers.APD_MCL_2DSlowScanView import APD_MCL_2DSlowScanView, APD_MCL_3DSlowScanView
app.load_view(APD_MCL_2DSlowScanView(app))
app.load_view(APD_MCL_3DSlowScanView(app))

from FoundryDataBrowser.viewers.WinSpecMCL2DSlowScanView import WinSpecMCL2DSlowScanView
app.load_view(WinSpecMCL2DSlowScanView(app))

from FoundryDataBrowser.viewers.WinSpecRemoteReadoutView import WinSpecRemoteReadoutView
app.load_view(WinSpecRemoteReadoutView(app))

from FoundryDataBrowser.viewers.power_scan_h5 import PowerScanH5View
app.load_view(PowerScanH5View(app))

from FoundryDataBrowser.viewers.sync_raster_scan_h5 import SyncRasterScanH5
app.load_view(SyncRasterScanH5(app))

from FoundryDataBrowser.viewers.auger_sync_raster_scan import AugerSyncRasterScanH5
app.load_view(AugerSyncRasterScanH5(app))

from FoundryDataBrowser.viewers.auger_spectrum_h5 import AugerSpectrumH5
app.load_view(AugerSpectrumH5(app))

from FoundryDataBrowser.viewers.auger_sync_raster_scan_h5 import AugerSyncRasterScanH5View
app.load_view(AugerSyncRasterScanH5View(app))

from FoundryDataBrowser.viewers.power_scan_npz import PowerScanNPZView
app.load_view(PowerScanNPZView(app))

sys.exit(app.exec_())