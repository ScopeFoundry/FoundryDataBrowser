# -*- mode: python -*-

block_cipher = None


a = Analysis(['foundry_data_browser.py'],
             pathex=['C:\\Users\\lab\\Documents\\foundry_scope'],
             binaries=[],
             datas=[('..\ScopeFoundry\data_browser.ui','ScopeFoundry'),
             		('viewers\*.py', 'viewers')],
             hiddenimports=["h5py.defs","h5py.utils","h5py.h5ac","h5py._proxy","scipy.signal","skimage", "skimage.feature", "pywt._extensions._cwt"],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='foundry_data_browser',
          debug=False,
          strip=False,
          upx=True,
          console=True )
