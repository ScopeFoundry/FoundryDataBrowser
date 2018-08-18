# -*- mode: python -*-

block_cipher = None


a = Analysis(['foundry_data_browser.py'],
             pathex=['/Users/esbarnard/Dropbox/MolecularFoundry/foundry_scope/FoundryDataBrowser'],
             binaries=[],
             datas=[],
             hiddenimports=['ipykernel.datapub', 'pywt._extensions._cwt', 'qtpy', 'pyqtgraph', 'scipy._lib.messagestream', 'scipy.misc.imread', 'matplotlib', 'skimage'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['tkinter',],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='foundry_data_browser',
          debug=False,
          strip=False,
          upx=True,
          console=True , icon='../ScopeFoundry/scopefoundry_logo2_1024.icns')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='foundry_data_browser')
app = BUNDLE(coll,
             name='foundry_data_browser.app',
             icon='../ScopeFoundry/scopefoundry_logo2_1024.icns',
             info_plist={
            'NSHighResolutionCapable': 'True'
            },
             bundle_identifier=None)
