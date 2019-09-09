from setuptools import setup

setup(
    name = 'FoundryDataBrowser',
    
    version = '190903.1',
    
    description = 'Data Browser for Foundry microscope data files',
    #long_description =open('README.md', 'r').read(), 
    
    # Author details
    author='Edward S. Barnard',
    author_email='esbarnard@lbl.gov',

    # Choose your license
    license='BSD',
    
    url='http://www.scopefoundry.org/',

    package_dir={'FoundryDataBrowser': '.'},
    
    packages=['FoundryDataBrowser', 'FoundryDataBrowser.viewers',],
    
    #packages=find_packages('.', exclude=['contrib', 'docs', 'tests']),
    #include_package_data=True,  
    
    package_data={
        '':["*.ui"], # include QT ui files 
        },
    
    #scripts=['bin/funniest-joke'],
    entry_points = {
        'console_scripts': ['FoundryDataBrowser=FoundryDataBrowser.foundry_data_browser:main'],
    },
    
    install_requires=[
          'ScopeFoundry',
          ],
    )
