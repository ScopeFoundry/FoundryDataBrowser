Foundry Data Browser
====================


A ScopeFoundry data browser that includes viewers for many of the experiement
data types at the Molecular Foundry

ScopeFoundry is a Python platform for controlling custom laboratory 
experiments and visualizing scientific data

<http://www.scopefoundry.org>

Maintainer
----------

Edward S. Barnard <esbarnard@lbl.gov>

Contributors
------------

Coming soon...


Requirements
------------

`ScopeFoundry` and its dependencies
	
  
Installation
------------

Using the anaconda python distribution we can set up an environment that works for the
Foundry Data Browser

```
$ conda create -n scopefoundry python=3.5
$ source activate scopefoundry
(scopefoundry) $ conda install numpy pyqt qtpy h5py
(scopefoundry) $ pip install pyqtgraph
(scopefoundry) $ pip install git+git://github.com/ScopeFoundry/ScopeFoundry.git
(scopefoundry) $ python foundry_data_browser.py
```