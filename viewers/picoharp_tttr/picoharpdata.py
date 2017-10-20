# Based on read_phd.m matlab reader

"""
Edward Barnard

2011-12-16 Initial version for PHD files v1.0 and v1.1
2012-01-25 Updated for PHD file v2.0
2012-10-03 Updated for handling PT2 files (TTTR mode)
2012-12-09 Added handy t2_event_ arrays
2013-11-20 (N. Borys) Fixed overflow problem due to implicit use of 32 bit
           integers for the T2WRAPAROUND value and t2_times class property.
"""


import struct
from struct import unpack
import numpy as np
from pprint import pprint
import os

def _read_bin(datfile,fmt):
    data_string = datfile.read(struct.calcsize(fmt))
    #print len(data_string)
    return struct.unpack(fmt, data_string)

def _seek_read(datfile,loc,fmt):
    datfile.seek(loc, os.SEEK_SET)
    data = _read_bin(datfile, fmt)
    if len(data) == 1:
        return data[0]
    else:
        return data

T2WRAPAROUND = np.int64(210698240)
#T2WRAPAROUND = 210698240        


class PicoHarpData(object):

    def __init__(self, filename, debug=False):
        
        self.filename = filename
        
        phdfile = self.phdfile = open(self.filename, 'rb')
        
        
        #### ASCII file header
        
        self.ident = _read_bin(phdfile,'< 16s')[0].strip('\x00')
        if debug: print("ident: ", repr(self.ident))
        
        self.format_version = _read_bin(phdfile,'< 6s')[0].strip('\x00')
        if debug: print("format_version:", repr(self.format_version))
        
        if self.format_version not in ['1.0','1.1', '2.0']:
            print('Warning: This program is for versions 1.0 and 1.1 or 2.0 only. Aborted.')
            return None
        
        if self.format_version in ['1.0','1.1']:
            self.V1 = True
            self.V2 = False
        elif self.format_version == '2.0':
            self.V1 = False
            self.V2 = True
        
        if debug: print("V1", self.V1, "V2", self.V2)
        
        self.creator_name = _read_bin(phdfile,'< 18s')[0].strip('\x00')
        if debug: print(repr(self.creator_name))
    
        self.creator_version = _read_bin(phdfile,'< 12s')[0].strip('\x00')
        if debug: print(repr(self.creator_version))
        
        self.file_time = _read_bin(phdfile,'< 18s')[0].strip('\x00')
        if debug: print("file_time: ", repr(self.file_time))
        
        CRLF = phdfile.read(2)
        
        self.comment = _read_bin(phdfile,'< 256s')[0].strip('\x00')
        if debug: print("comment :", repr(self.comment))
    
        #### Binary file header
    
        # has 18 int32's 
        #binaryfile_header = phdfile.read(18*4)
        binaryfile_header = _read_bin(phdfile,'< 18i') #unpack('< 18i', binaryfile_header)
        if debug: print("binaryfile_header: ", repr(binaryfile_header))
        
        #  parse / unpack binaryfile_header
        (self.number_of_curves, self.bits_per_histo_bin,
         self.routing_channels, self.number_of_boards,
         self.active_curve,     self.measurement_mode,
         self.sub_mode,         self.range_no,
         self.offset,           self.tacq,
         self.stop_at,          self.stop_on_ovf,
         self.restart,          self.disp_lin_log,
         self.disp_time_axis_from, self.disp_time_axis_to,
         self.disp_count_axis_from, self.disp_count_axis_to) = binaryfile_header
        
        if debug:
            print("number_of_curves", self.number_of_curves)
            print("number_of_boards", self.number_of_boards)
        
        
        
        ####
        
        self.disp_curve_map_to = np.zeros(8, int)
        self.disp_curve_show = np.zeros(8, bool)
        
        for i in range(8):
            self.disp_curve_map_to[i], = _read_bin(phdfile,'<i')  #unpack('<i', phdfile.read(4))
            self.disp_curve_show[i],   = _read_bin(phdfile,'<i')  #unpack('<i', phdfile.read(4))   
    
        if debug: print(repr(self.disp_curve_map_to))
        if debug: print(repr(self.disp_curve_show))
        
        ####

        self.param_start = [0,0,0]
        self.param_step = [0,0,0]
        self.param_end = [0,0,0]
        
        for i in [0,1,2]:
            self.param_start[i], =  _read_bin(phdfile,'<f') #unpack('<f', phdfile.read(4))
            self.param_step[i], =  _read_bin(phdfile,'<f') #unpack('<f', phdfile.read(4))
            self.param_end[i], =  _read_bin(phdfile,'<f') #unpack('<f', phdfile.read(4))
     
        ####
     
        (self.repeat_mode, self.repeats_per_curve, 
            self.repeat_time, self.repeat_wait_time) =  _read_bin(phdfile,'<4i') ##unpack('< 4i', phdfile.read(4*4))

        self.script_name = _read_bin(phdfile,'< 20s')[0].strip('\x00') #unpack('< 20s', phdfile.read(20))[0].strip('\x00')
    
    
        #### Header for each board
        
        self.boards = []
        
        for board_n in range(self.number_of_boards):
            if debug: print("board", board_n)
            board = PicoHarpBoard()
            board.hardware_ident = _read_bin(phdfile,'< 16s')[0].strip('\x00') #unpack('< 16s', phdfile.read(16))[0].strip('\x00')
            board.hardware_version = _read_bin(phdfile,'< 8s')[0].strip('\x00') #unpack('< 8s', phdfile.read(8))[0].strip('\x00')
            ( board.hardware_serial, board.sync_divider,
              board.cfd_zero_cross_0, board.cfd_level_0,
              board.cfd_zero_cross_1, board.cfd_level_1  ) = _read_bin(phdfile,'< 6i') #unpack('< 6i', phdfile.read(6*4))
              
            board.resolution, = _read_bin(phdfile,'<f') #unpack('<f', phdfile.read(4))
            
            if debug: print("board", board_n, "resolution", board.resolution) 
            
            if self.V2:
                (RouterModelCode, board.RouterEnabled) = _read_bin(phdfile, '< i i')
                
                (   board.RtChan1_InputType,
                    board.RtChan1_InputLevel,
                    board.RtChan1_InputEdge,
                    board.RtChan1_CFDPresent,
                    board.RtChan1_CFDLevel,
                    board.RtChan1_CFDZeroCross) = _read_bin(phdfile, '< 6i')
                (   board.RtChan2_InputType,
                    board.RtChan2_InputLevel,
                    board.RtChan2_InputEdge,
                    board.RtChan2_CFDPresent,
                    board.RtChan2_CFDLevel,
                    board.RtChan2_CFDZeroCross) = _read_bin(phdfile, '< 6i')
                (   board.RtChan3_InputType,
                    board.RtChan3_InputLevel,
                    board.RtChan3_InputEdge,
                    board.RtChan3_CFDPresent,
                    board.RtChan3_CFDLevel,
                    board.RtChan3_CFDZeroCross) = _read_bin(phdfile, '< 6i')
                (   board.RtChan4_InputType,
                    board.RtChan4_InputLevel,
                    board.RtChan4_InputEdge,
                    board.RtChan4_CFDPresent,
                    board.RtChan4_CFDLevel,
                    board.RtChan4_CFDZeroCross) = _read_bin(phdfile, '< 6i')                    
            self.boards.append(board)
        
        #### Header for each histogram (curve)
        
        if self.measurement_mode == 0:
            self.curves = []
                    
            for curve_n in range(self.number_of_curves):
                if debug: print("curve", curve_n, "reading header")
                c = PicoHarpCurve()
                
                c.index, c.time_of_recording = _read_bin(phdfile, '< i I')
                
                c.hardware_ident = _read_bin(phdfile, '<16s')[0].strip('\x00')
                c.hardware_version = _read_bin(phdfile, '<8s')[0].strip('\x00')
                c.hardware_serial, = _read_bin(phdfile, '< i')
    
                c.hardware_ident =  c.hardware_ident.strip('\x00')
                c.hardware_version =  c.hardware_version.strip('\x00')
    
                (c.sync_divider, c.cfd_zero_cross_0, c.cfd_level_0,
                 c.cfd_zero_cross_1, c.cfd_level_1, c.offset, c.routing_channel, c.ext_devices,
                 c.meas_mode, c.sub_mode) = _read_bin(phdfile, '< 10i')
                
                
                c.P1, c.P2, c.P3 = _read_bin(phdfile, '< 3f')
                
                c.range_no, c.resolution, c.channels, c.tacq = _read_bin(phdfile, '< i f i i')
                
                if debug: print("curve", curve_n,"resolution", c.resolution)
                
                
                c.stop_after, c.stop_reason = _read_bin(phdfile, '< i i')
                
                c.inp_rate_0, c.inp_rate_1 = _read_bin(phdfile, '< i i')
                
                c.hist_count_rate, c.integral_count = _read_bin(phdfile, '< i l')
                
                c.reserved, c.data_offset = _read_bin(phdfile, '< i i')
                
                if self.V2:
                    (   c.RouterModelCode,
                        c.RouterEnabled,
                        c.RtChan_InputType,
                        c.RtChan_InputLevel,
                        c.RtChan_InputEdge,
                        c.RtChan_CFDPresent,
                        c.RtChan_CFDLevel,
                        c.RtChan_CFDZeroCross) = _read_bin(phdfile, '< 8i')
                
                self.curves.append(c)
            
            #### Read all histograms into curves[n].data
            
            for curve_n in range(self.number_of_curves):
                curve = self.curves[curve_n]
                #curve.data = _seek_read(phdfile, curve.data_offset, '< %iI' % curve.channels) #uint32
                if debug: print('curve', curve_n, 'seek to ', curve.data_offset)
                phdfile.seek(curve.data_offset, os.SEEK_SET)
                if debug: print('curve', curve_n, 'read', curve.channels)
                raw_binary_string = phdfile.read(curve.channels*4)
                if debug: print('curve', curve_n, 'convert to numpy')
                curve.data = np.fromstring(raw_binary_string, dtype=np.uint32, count=curve.channels)
                if debug: print('curve', curve_n, 'done')
                curve.time = np.arange(0, curve.resolution*curve.channels, curve.resolution) 

        elif self.measurement_mode == 2:
        
            #### Header for T2/T3 Mode
            
            self.ext_devices = _read_bin(phdfile, '< i')[0]
            reserved = _read_bin(phdfile, '< 2i')
            self.inp_rate0, self.inp_rate1 = _read_bin(phdfile, '< 2i')            
            self.stop_after = _read_bin(phdfile, '<i')[0] # stooped after this many ms
            self.stop_reason = _read_bin(phdfile, '<i')[0]
            self.stop_reason_name = ['timeover', 'manual', 'overflow'][self.stop_reason]
            self.num_records = _read_bin(phdfile, '<i')[0]
            self.img_hdr_size = _read_bin(phdfile, '<i')[0]
            
            if self.img_hdr_size > 0:
                self.img_hdr = _read_bin(phdfile, '< %i I' % self.img_hdr_size)
                
            if debug: print('T2 records read', self.num_records)
            raw_binary_string = phdfile.read(self.num_records*4)  #uint32's (4 bytes each)
            if debug: print('T2 records convert to numpy')
            self.t2_rawdata = np.fromstring(raw_binary_string, dtype=np.uint32, count=self.num_records)
            
            self.t2_channels = self.t2_rawdata >> 28
            self.t2_times_wrapped = self.t2_rawdata & (~( 0b1111 << 28 ))
            
            self.t2_special_record_mask = self.t2_channels == 0b1111
            self.t2_standard_record_mask = np.logical_not(self.t2_special_record_mask)

            self.t2_overflow_mask       = self.t2_special_record_mask * ( (self.t2_times_wrapped & 0b1111) == 0 )
            self.t2_overflow_index = self.t2_overflow_mask.nonzero()[0]

            #unwrap times using overflows
            self.t2_times = np.zeros(self.num_records, dtype=np.int64)
            #self.t2_times[:] = self.t2_times_wrapped[:]
            
            #for n in range(len(self.t2_overflow_index)-1):
            #    i0,i1 = self.t2_overflow_index[n:n+2]
            #    self.t2_times[i0:i1] += n*T2WRAPAROUND
            #self.t2_times[self.t2_overflow_index[-1]:] += T2WRAPAROUND * (len(self.t2_overflow_index) - 1)
            
            self.t2_times = self.t2_times_wrapped + np.cumsum(self.t2_overflow_mask, dtype=np.int64) * T2WRAPAROUND
            
            
            self.t2_times_c0 = self.t2_times[ self.t2_channels == 0 ]
            self.t2_times_c1 = self.t2_times[ self.t2_channels == 1 ]
            
            # handy event arrays
            self.t2_event_times = self.t2_times[np.nonzero(self.t2_standard_record_mask)]
            self.t2_event_channels = self.t2_channels[ np.nonzero(self.t2_standard_record_mask) ]

            
        else:
            print("measurement_mode %i is not supported / unknown" % self.measurement_mode)
        
        if debug: pprint(self.__dict__ )
    
class PicoHarpBoard(object):
    
    def __repr__(self):
        return "PicoHarpBoard" + dict.__repr__(self.__dict__)

class PicoHarpCurve(object):
    
    def __repr__(self):
        return "PicoHarpCurve" + dict.__repr__(self.__dict__)

if __name__ == '__main__':
    #phd = PicoHarpData('cdse-singlextal-0001-mtixtl-lifetime.phd')
    phd = PicoHarpData('cigs-7b-loc2-lifetime.phd',debug=True)
    
    pprint(phd.boards[0].__dict__)
    pprint(phd.curves[0].__dict__)
    
    import pylab as pl
    
    print(phd.curves[0].time.shape)
    print(phd.curves[0].data.shape)
    pl.semilogy(phd.curves[0].time, phd.curves[0].data)
    
    pl.show()
