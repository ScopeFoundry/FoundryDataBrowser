# Build with `python setup.py build_exe`
from cx_Freeze import setup, Executable

import shutil
from glob import glob
# Remove the build folder
shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("dist", ignore_errors=True)
import sys

import os
#os.environ["TCL_LIBRARY"] = ""
#os.environ["TK_LIBRARY"] = ""
sys.path.append('..')

includes = ['PyQt5.QtCore', 'PyQt5.QtWidgets', 'sip', 'pyqtgraph.graphicsItems', 'pyqtgraph.debug','pyqtgraph.ThreadsafeTimer', 'qtconsole',
            'numpy', 'atexit', 'ScopeFoundry', 'numpy.core._methods', 'numpy.lib.format']
excludes = ['cvxopt','_gtkagg', '_tkagg', 'bsddb', 'curses', 'email', 'pywin.debugger',
    'pywin.debugger.dbgcon', 'pywin.dialogs', 'tcl','tables',
    'Tkconstants', 'Tkinter','tkinter', 'zmq','PySide','pysideuic','scipy','matplotlib']

if sys.version[0] == '2':
    # causes syntax error on py2
    excludes.append('PyQt4.uic.port_v3')

include_files = []

base = None
if sys.platform == "win32":
    include_files = [r'C:\Users\lab\Anaconda3\Library\bin\libEGL.dll']
    base = "Win32GUI"

build_exe_options = {
    'excludes': excludes,
    'includes':includes,
    'include_msvcr':True,
    'include_files':include_files,}#
#    'compressed':True, 'copy_dependent_files':True, 'create_shared_zip':True,
#    'include_in_shared_zip':True, 'optimize':2}

setup(name = "cx_freeze plot test",
      version = "0.1",
      description = "cx_freeze plot test",
      options = {"build_exe": build_exe_options},
      executables = [Executable("foundry_data_browser.py", base=base)])
