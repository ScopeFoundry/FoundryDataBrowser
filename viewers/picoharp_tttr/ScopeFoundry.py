import numpy as np
import UsefulUtils as uu
from UsefulUtils import Bunch

import h5py

def load_andor_ccd_readout(filename, description=''):
    
    d = dict(np.load(filename))
    
    #uu.copy_string('\n'.join(d.keys()))
    
    o = Bunch(data_dict = d,
              keys = d.keys(),
              filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description,
              wavelength = d['wls'],
              intg_time = d['andor_ccd_exposure_time'],
              intensity = d['spectrum'][0],
              center_wl = d['acton_spectrometer_center_wl']              
              )
             
    o.everything=Bunch()
    for key in o.keys:
        setattr(o.everything, key, d[key])
    
    return o
    
def load_apd_map(filename, description=""):
    d = dict(np.load(filename))
    
    #uu.copy_string('\n'.join(d.keys()))
    
    o = Bunch(data_dict = d,
              keys = d.keys(),
              filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description,
              intensity = d['count_rate_map'],
              extent = d['imshow_extent']
             )
    o.everything = Bunch()
    for key in o.keys:
        setattr(o.everything, key, d[key])
    
    return o

    
def load_picoharp_transient(filename, description="", truncate_bins = False):
    d = dict(np.load(filename))
    
    #uu.copy_string('\n'.join(d.keys()))
    
    o = Bunch(data_dict = d,
              keys = d.keys(),
              filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description,
              intensity = d['time_histogram'],
              time = d['time_array'],
              resolution = d['picoharp_Resolution'],
              rep_rate = d['picoharp_count_rate0']
             )
    o.num_bins = int(np.ceil((1/o.rep_rate)/(o.resolution*1e-12)))
    if truncate_bins:
        o.time = o.time[0:o.num_bins]
        o.intensity = o.intensity[0:o.num_bins]
    
    o.everything = Bunch()
    for key in o.keys:
        setattr(o.everything, key, d[key])
    
    return o
    
def load_trpl_scan(filename, description):
    d = dict(np.load(filename))
    
    #uu.copy_string('\n'.join(d.keys()))
    
    o = Bunch(data_dict = d,
              keys = d.keys(),
              filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description,
              extent = d['imshow_extent'],
              time = d['time_array'],
              resolution = d['picoharp_Resolution'],
              transients = d['time_trace_map'],
              rows = d['Nv'],
              cols = d['Nh']
             )
    
    o.everything = Bunch()
    for key in o.keys:
        setattr(o.everything, key, d[key])
    
    # average transient
    o.avg_transient = np.average(o.transients, axis=(0,1))
    
    # integrated intensity map
    o.intg_int_map = np.zeros((o.rows, o.cols), dtype=float)
    for row in range(o.rows):
        for col in range(o.cols):
            o.intg_int_map[row,col] = np.sum(o.transients[row,col])
    
    return o
    
def load_spec_and_glue(filename, description=""):
    h5_f = h5py.File(filename, 'r')
    
    o = Bunch(
      centers = h5_f['andor_ccd_step_and_glue']['center_wl_actual'][()].copy(),
      single_specs =  h5_f['andor_ccd_step_and_glue']['spectra_data'][()].copy(),
      single_wavelengths = h5_f['andor_ccd_step_and_glue']['wls'][()].copy(),
      filename = filename,
      filetitle = uu.get_file_title(filename),
      description=description
    )
    
    # Initialize with the first spectrum
    wls = o.single_wavelengths[0][:]
    raw_ints = o.single_specs[0][0,:]
    
    # Progressively merge spectra into the lists of wls.
    for i in range(1, o.centers.shape[0]):
        # Determine the array indices and size of the regions of overlap
        i_overlap_start = np.searchsorted(wls, o.single_wavelengths[i][0])
        overlap_size_1 = wls.shape[0] - i_overlap_start
        overlap_size_2 = np.searchsorted(o.single_wavelengths[i], wls[-1])
        #print(i_overlap_start, overlap_size_1, overlap_size_2)
        
        # Generate a uniformly spaced points in the overlap region 
        wls_overlap = np.linspace(o.single_wavelengths[i][0], wls[-1], 
                                  (overlap_size_1+overlap_size_2)/2)
        
        # Take the average of the interpolated values from the two data sets in the
        # interpolated region.
        raw_int_overlap = (np.interp(wls_overlap, wls[i_overlap_start:], raw_ints[i_overlap_start:] ) + 
                           np.interp(wls_overlap, o.single_wavelengths[i][0:overlap_size_2], 
                                     o.single_specs[i][0,0:overlap_size_2]))/2.0
               
        # Append the averaged overlapped data to the non-overlapped data at the beginning.
        wls = np.append(wls[0:i_overlap_start], wls_overlap)
        raw_ints = np.append(raw_ints[0:i_overlap_start], raw_int_overlap)
        
        # Append the remaining non-overlapped data 
        wls = np.append(wls, o.single_wavelengths[i][overlap_size_2:])
        raw_ints = np.append(raw_ints, o.single_specs[i][0,overlap_size_2:])
        
    #print wls
    o.wavelength = wls
    o.intensity = raw_ints
    
    h5_f.close()
    
    return o

def load_kinetic_spectra(filename, description=''):
    
    d = dict(np.load(filename))
    
    #uu.copy_string('\n'.join(d.keys()))
    
    o = Bunch(data_dict = d,
              keys = d.keys(),
              filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description,
              wavelength = d['wls'],
              intg_time = d['andor_ccd_exposure_time'],
              center_wl = d['acton_spectrometer_center_wl'],
              times = d['start_times'],
              frame_count = int(d['kinetic_spectra_frames']),
              specs = d['kinetic_spectra']
              )
             
    o.everything=Bunch()
    for key in o.keys:
        setattr(o.everything, key, d[key])
    
    return o

def load_spec_scan(filename, description=""):
    h5_f = h5py.File(filename, 'r')
    
    o = Bunch(
      specs = h5_f['spec_scan']['spec_map'][()].copy(),
      wavelength = h5_f['spec_scan']['wls'][()].copy(),
      extent = h5_f['spec_scan'].attrs['imshow_extent'][()].copy(),
      filename = filename,
        filetitle = uu.get_file_title(filename),
        description=description
    )
    
    return o

def merge_step_and_glue_specs(centers, wavelengths, specs):
    # Initialize with the first spectrum
    wls = wavelengths[0][:]
    raw_ints = specs[0][0,:]

    # Progressively merge spectra into the lists of wls.
    for i in range(1, centers.shape[0]):
        # Determine the array indices and size of the regions of overlap
        i_overlap_start = np.searchsorted(wls, wavelengths[i][0])
        overlap_size_1 = wls.shape[0] - i_overlap_start
        overlap_size_2 = np.searchsorted(wavelengths[i], wls[-1])
        #print(i_overlap_start, overlap_size_1, overlap_size_2)

        # Generate a uniformly spaced points in the overlap region 
        wls_overlap = np.linspace(wavelengths[i][0], wls[-1], 
                                  (overlap_size_1+overlap_size_2)/2)

        # Take the average of the interpolated values from the two data sets in the
        # interpolated region.
        raw_int_overlap = (np.interp(wls_overlap, wls[i_overlap_start:], raw_ints[i_overlap_start:] ) + 
                           np.interp(wls_overlap, wavelengths[i][0:overlap_size_2], 
                                     specs[i][0,0:overlap_size_2]))/2.0

        # Append the averaged overlapped data to the non-overlapped data at the beginning.
        wls = np.append(wls[0:i_overlap_start], wls_overlap)
        raw_ints = np.append(raw_ints[0:i_overlap_start], raw_int_overlap)

        # Append the remaining non-overlapped data 
        wls = np.append(wls, wavelengths[i][overlap_size_2:])
        raw_ints = np.append(raw_ints, specs[i][0,overlap_size_2:])

    return wls, raw_ints

def load_power_sweep(filename, description=''):

    d = dict(np.load(filename))

    uu.copy_string('\n'.join(d.keys()))

    o = Bunch(data_dict = d,
              keys = d.keys(),
              filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description,
              powers = d['power_meter_power'], 
              N = d['Np']
              )

    if d['power_scan_motorized_collect_apd']:
        raise NotImplementedError('APD power scan not implemented')
    
    if d['power_scan_motorized_collect_lifetime']:
        o.rep_rate = d['picoharp_count_rate0']
        o.bin_size = d['picoharp_Resolution']
        o.num_bins = int(np.ceil((1/o.rep_rate)/(o.bin_size*1e-12)))
        o.time = d['time_array'][0:o.num_bins] # already in ns
        o.intensity = d['time_traces'][:,0:o.num_bins]
    
    if d['power_scan_motorized_collect_lockin']:
        raise NotImplementedError('Lockin power scan not implemented')
        
    if d['power_scan_motorized_collect_spectrum']:
        o.wavelength = d['wls']
        o.intg_time = d['andor_ccd_exposure_time']
        o.center_wl = d['acton_spectrometer_center_wl']
        o.specs = d['spectra']
        o.intg_spectrum_ints = np.sum(o.specs, axis=1)
        
    if d['power_scan_motorized_collect_step_and_glue']:
        N = d['Np']
        for i in range(N):
            wls, ints = merge_step_and_glue_specs(d['step_and_glue_centers'],
                                                  d['step_and_glue_wavelengths'][i],
                                                  d['step_and_glue_specs'][i])
            if i == 0:
                o.specs = np.empty([N, wls.shape[0]], dtype=float)
                o.wavelength = wls
            o.specs[i] = ints
        
        o.intg_time = d['andor_ccd_exposure_time']       
        o.centers = d['step_and_glue_centers'][0]
    
    o.everything=Bunch()
    for key in o.keys:
        setattr(o.everything, key, d[key])

    return o
    
def load_spec_scan(filename, description=""):

    d = dict(np.load(filename))
    
    uu.copy_string('\n'.join(d.keys()))
    
    o = Bunch(data_dict = d,
              keys = d.keys(),
              filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description,
              wavelength = d['wls'],
              intg_time = d['andor_ccd_exposure_time'],
              center_wl = d['acton_spectrometer_center_wl'],
              extent = d['imshow_extent'],
              rows = d['Nv'],
              cols = d['Nh'],
              specs = d['spec_map'],
             )
             
    o.everything=Bunch()
    for key in o.keys:
        setattr(o.everything, key, d[key])
    
    return o

    