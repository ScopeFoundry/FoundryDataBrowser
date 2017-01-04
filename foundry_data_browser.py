from __future__ import absolute_import
from ScopeFoundry.data_browser import DataBrowser


import sys

app = DataBrowser(sys.argv)


# views are loaded in order of more generic to more specific.
## ie the last loaded views are checked first for compatibility

from viewers.images import ScipyImreadView
app.load_view(ScipyImreadView(app))

from viewers.apd_confocal_npz import ApdConfocalNPZView
app.load_view(ApdConfocalNPZView(app))

from viewers.picoharp_npz import PicoHarpNPZView
app.load_view(PicoHarpNPZView(app))

from viewers.hyperspec_npz import HyperSpecNPZView
app.load_view(HyperSpecNPZView(app))

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


sys.exit(app.exec_())