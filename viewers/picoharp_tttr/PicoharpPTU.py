import struct
from struct import unpack
import numpy as np
from pprint import pprint
import os

def read_bin(datfile,fmt):
    data_string = datfile.read(struct.calcsize(fmt))
    #print len(data_string)
    return struct.unpack(fmt, data_string)

class PicoHarpPTU(object):

    def __init__(self, filename, debug=False):
        self.debug = debug 
        self.load(filename)
        
    def load(self, filename):        
        # Constants for PTU Header
        tyEmpty8      = struct.unpack('!I', 'FFFF0008'.decode('hex'))[0]
        tyBool8       = struct.unpack('!I', '00000008'.decode('hex'))[0]
        tyInt8        = struct.unpack('!I', '10000008'.decode('hex'))[0]
        tyBitSet64    = struct.unpack('!I', '11000008'.decode('hex'))[0]
        tyColor8      = struct.unpack('!I', '12000008'.decode('hex'))[0]
        tyFloat8      = struct.unpack('!I', '20000008'.decode('hex'))[0]
        tyDateTime   =  struct.unpack('!I', '21000008'.decode('hex'))[0]
        tyFloat8Array = struct.unpack('!I', '2001FFFF'.decode('hex'))[0]
        tyAnsiString  = struct.unpack('!I', '4001FFFF'.decode('hex'))[0]
        tyWideString  = struct.unpack('!I', '4002FFFF'.decode('hex'))[0]
        tyBinaryBlob  = struct.unpack('!I', 'FFFFFFFF'.decode('hex'))[0]
        print(struct.unpack('!I', 'FFFF0008'.decode('hex'))[0])
        
        # RecordTypes
        rtPicoHarpT3     = struct.unpack('!I', '00010303'.decode('hex'))[0] # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $03 (PicoHarp)
        rtPicoHarpT2     = struct.unpack('!I', '00010203'.decode('hex'))[0] # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $03 (PicoHarp)
        rtHydraHarpT3    = struct.unpack('!I', '00010304'.decode('hex'))[0] # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $04 (HydraHarp)
        rtHydraHarpT2    = struct.unpack('!I', '00010204'.decode('hex'))[0] # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $04 (HydraHarp)
        rtHydraHarp2T3   = struct.unpack('!I', '01010304'.decode('hex'))[0] # (SubID = $01 ,RecFmt: $01) (V2), T-Mode: $03 (T3), HW: $04 (HydraHarp)
        rtHydraHarp2T2   = struct.unpack('!I', '01010204'.decode('hex'))[0] # (SubID = $01 ,RecFmt: $01) (V2), T-Mode: $02 (T2), HW: $04 (HydraHarp)
        rtTimeHarp260NT3 = struct.unpack('!I', '00010305'.decode('hex'))[0] # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $05 (TimeHarp260N)
        rtTimeHarp260NT2 = struct.unpack('!I', '00010205'.decode('hex'))[0] # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $05 (TimeHarp260N)
        rtTimeHarp260PT3 = struct.unpack('!I', '00010306'.decode('hex'))[0] # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $06 (TimeHarp260P)
        rtTimeHarp260PT2 = struct.unpack('!I', '00010206'.decode('hex'))[0] # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $06 (TimeHarp260P)

        self.filename = filename 
        self.ptu_file = ptu_file = open(filename, 'rb')
        
        #Magic = read_bin(ptu_file, 8, '*char');
        Magic = read_bin(ptu_file, '< 8s')[0].strip('\x00')
         
        if Magic != 'PQTTTR':
            print('Magic invalid, this is not a PTU file.')
    
        #Version = fread(fid, 8, '*char');
        Version = read_bin(ptu_file, '< 8s')[0].strip('\x00')
        print(Version)
    
        # Load the PTU Header
        i = 0
        TagIdent = ''
        self.header_tag_dict = dict()
        while TagIdent != 'Header_End':
            #read Tag Head
            #TagIdent = fread(fid, 32, '*char'); % TagHead.Ident
            TagIdent = read_bin(ptu_file, '< 32s')[0].strip('\x00')
            #TagIdx = fread(fid, 1, 'int32');    % TagHead.Idx
            TagIdx = read_bin(ptu_file, '<i')[0]
            #TagTyp = fread(fid, 1, 'uint32');  
            TagTyp = read_bin(ptu_file, '<I')[0]
            
            if TagIdx > -1:
                EvalName = str.format('%s (%d)', TagIdent, TagIdx) #[TagIdent '(' int2str(TagIdx + 1) ')'];
            else:
                EvalName = TagIdent;
            
            #check Typ of Header
            if TagTyp == tyEmpty8:
                #print 'Empty'
                read_bin(ptu_file, '<q')[0]  # Just read, don't save anything
                tag_value = None 
    
            elif TagTyp == tyBool8:
                #print 'bool8'
                temp = read_bin(ptu_file, '<q')[0]
                if temp == 0:
                    tag_value = False 
                else:
                    tag_value = True
               
            elif TagTyp == tyInt8:
                #print 'int8'
                tag_value = read_bin(ptu_file, '<q')[0]
    
            elif TagTyp == tyBitSet64:
                #print 'bitSet64'
                tag_value = read_bin(ptu_file, '<q')[0]
    
            elif TagTyp == tyColor8:
                #print 'Color8'
                tag_value = read_bin(ptu_file, '<q')[0]
    
            elif TagTyp == tyFloat8:
                #print 'Float8'
                tag_value = read_bin(ptu_file, '<d')[0]
    
            elif TagTyp == tyFloat8Array:
                #print 'Float8Array'
                tag_value = read_bin(ptu_file, '<q')[0]
                ptu_file.seek(tag_value, whence=1)
    
            elif TagTyp == tyDateTime:
                #print 'DateTime'
                tag_value = read_bin(ptu_file, '<d')[0]
    
            elif TagTyp == tyAnsiString:
                #print 'AnsiString'
                tag_size = read_bin(ptu_file, '<q')[0]
                tag_value = read_bin(ptu_file, '< %ds' % tag_size)[0].strip('\x00')
    
            elif TagTyp == tyWideString:
                print('WideString header fields not supported!')
                raise(NotImplementedError)            
                #             case tyWideString 
                #                 % Matlab does not support Widestrings at all, just read and
                #                 % remove the 0's (up to current (2012))
                #                 TagInt = fread(fid, 1, 'int64');
                #                 TagString = fread(fid, TagInt, '*char');
                #                 TagString = (TagString(TagString ~= 0))';
                #                 fprintf(1, '%s', TagString);
                #                 if TagIdx > -1
                #                    EvalName = [TagIdent '(' int2str(TagIdx + 1) ',:)'];
                #                 end;
                #                 eval([EvalName '=TagString;']);
                
            elif TagTyp == tyBinaryBlob:
                #print 'BinaryBlob'
                tag_size = read_bin(ptu_file, '<q')[0]
                tag_value = tag_size
                ptu_file.seek(tag_size, whence=1)
    
            else:
                print('Unknown Tag Type -- cannot process file.  Type Type=%d' % TagTyp)
                #print('Unknown Tag', TagTyp)
            
            self.header_tag_dict[EvalName] = tag_value
            
            # Keep track of how many fields we loaded.
            i = i + 1
    
        # End of reading the header
        print('Header successfully loaded.  %d tags loaded.' % i)
    
    
        res_fmt_rec_type = self.header_tag_dict['TTResultFormat_TTTRRecType']
        if res_fmt_rec_type == rtPicoHarpT3:
            print('rtPicoHarpT3')
            self.isT2 = False
        elif res_fmt_rec_type == rtPicoHarpT2:
            print('rtPicoHarpT2')
            self.isT2 = True
        elif res_fmt_rec_type == rtHydraHarpT3:
            print('rtHydraHarpT3')
            self.isT2 = False
        elif res_fmt_rec_type == rtHydraHarpT2:
            print('rtHydraHarpT2')
            self.isT2=True 
        elif res_fmt_rec_type == rtHydraHarp2T3:
            print('rtHydraHarp2T3')
            self.isT2 = True
        elif res_fmt_rec_type == rtHydraHarp2T2:
            print('rtHydraHarp2T2')
            self.isT2 = True
        elif res_fmt_rec_type == rtTimeHarp260NT3:
            print('rtTimeHarp260NT3')
            self.isT2 = False
        elif res_fmt_rec_type == rtTimeHarp260NT2:
            print('rtTimeHarp260NT2')
            self.isT2 = True
        elif res_fmt_rec_type == rtTimeHarp260PT3:
            print('rtTimeHarp260PT3')
            self.isT2 = False 
        elif res_fmt_rec_type == rtTimeHarp260PT2:
            print('rtTimeHarp260PT2')
            self.isT2 = True
        else:
            print('Unknown record type! RecordType = %d' % res_fmt_rec_type )

        if res_fmt_rec_type == rtPicoHarpT2:
            self.ReadPT2()
        else:
            print('Cannot read this record type!')
            raise(NotImplementedError)
        
    # %% Read PicoHarp T2
    def ReadPT2(self):
        WRAPAROUND=210698240
        
        self.num_records = self.header_tag_dict['TTResult_NumberOfRecords']
        
        raw_binary_string = self.ptu_file.read(self.num_records*4)  #uint32's (4 bytes each)

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
        
        self.t2_times = self.t2_times_wrapped + np.cumsum(self.t2_overflow_mask, dtype=np.int64) * WRAPAROUND
        
        
        self.t2_times_c0 = self.t2_times[ self.t2_channels == 0 ]
        self.t2_times_c1 = self.t2_times[ self.t2_channels == 1 ]
        
        # handy event arrays
        self.t2_event_times = self.t2_times[np.nonzero(self.t2_standard_record_mask)]
        self.t2_event_channels = self.t2_channels[ np.nonzero(self.t2_standard_record_mask) ]
        
        print('Loaded %d records' % self.num_records)
        


# 
# %% Got Photon
# %    TimeTag: Raw TimeTag from Record * Globalresolution = Real Time arrival of Photon
# %    DTime: Arrival time of Photon after last Sync event (T3 only) DTime * Resolution = Real time arrival of Photon after last Sync event
# %    Channel: Channel the Photon arrived (0 = Sync channel for T2 measurements)
# function GotPhoton(TimeTag, Channel, DTime)
#   global isT2;
#   global fpout;
#   global RecNum;
#   global MeasDesc_GlobalResolution;
#   global cnt_ph;
#   cnt_ph = cnt_ph + 1;
#   if(isT2)
#       fprintf(fpout,'%i CHN %1x %i %e\n', RecNum, Channel, TimeTag, (TimeTag * MeasDesc_GlobalResolution * 1e12));
#   else
#       fprintf(fpout,'%i CHN %1x %i %e %i\n', RecNum, Channel, TimeTag, (TimeTag * MeasDesc_GlobalResolution * 1e9), DTime);
#   end;
# end
# 
# %% Got Marker
# %    TimeTag: Raw TimeTag from Record * Globalresolution = Real Time arrival of Photon
# %    Markers: Bitfield of arrived Markers, different markers can arrive at same time (same record)
# function GotMarker(TimeTag, Markers)
#   global fpout;
#   global RecNum;
#   global cnt_ma;
#   cnt_ma = cnt_ma + 1;
#   fprintf(fpout,'%i MAR %x %i\n', RecNum, Markers, TimeTag);
# end
# 
# %% Got Overflow
# %  Count: Some TCSPC provide Overflow compression = if no Photons between overflow you get one record for multiple Overflows
# function GotOverflow(Count)
#   global fpout;
#   global RecNum;
#   global cnt_ov;
#   cnt_ov = cnt_ov + Count;
#   fprintf(fpout,'%i OFL * %x\n', RecNum, Count);
# end
# 
# %% Decoder functions
# 


# 
# %% Read HydraHarp/TimeHarp260 T3
# function ReadHT3(Version)
#     global fid;
#     global RecNum;
#     global TTResult_NumberOfRecords; % Number of TTTR Records in the File
#     OverflowCorrection = 0;
#     T3WRAPAROUND = 1024;
# 
#     for i = 1:TTResult_NumberOfRecords
#         RecNum = i;
#         T3Record = fread(fid, 1, 'ubit32');     % all 32 bits:
#         %   +-------------------------------+  +-------------------------------+ 
#         %   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|
#         %   +-------------------------------+  +-------------------------------+  
#         nsync = bitand(T3Record,1023);       % the lowest 10 bits:
#         %   +-------------------------------+  +-------------------------------+ 
#         %   | | | | | | | | | | | | | | | | |  | | | | | | |x|x|x|x|x|x|x|x|x|x|
#         %   +-------------------------------+  +-------------------------------+  
#         dtime = bitand(bitshift(T3Record,-10),32767);   % the next 15 bits:
#         %   the dtime unit depends on "Resolution" that can be obtained from header
#         %   +-------------------------------+  +-------------------------------+ 
#         %   | | | | | | | |x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x| | | | | | | | | | |
#         %   +-------------------------------+  +-------------------------------+
#         channel = bitand(bitshift(T3Record,-25),63);   % the next 6 bits:
#         %   +-------------------------------+  +-------------------------------+ 
#         %   | |x|x|x|x|x|x| | | | | | | | | |  | | | | | | | | | | | | | | | | |
#         %   +-------------------------------+  +-------------------------------+
#         special = bitand(bitshift(T3Record,-31),1);   % the last bit:
#         %   +-------------------------------+  +-------------------------------+ 
#         %   |x| | | | | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |
#         %   +-------------------------------+  +-------------------------------+ 
#         if special == 0   % this means a regular input channel
#            true_nSync = OverflowCorrection + nsync;
#            %  one nsync time unit equals to "syncperiod" which can be
#            %  calculated from "SyncRate"
#            GotPhoton(true_nSync, channel, dtime);
#         else    % this means we have a special record
#             if channel == 63  % overflow of nsync occured
#               if (nsync == 0) || (Version == 1) % if nsync is zero it is an old style single oferflow or old Version
#                 OverflowCorrection = OverflowCorrection + T3WRAPAROUND;
#                 GotOverflow(1);
#               else         % otherwise nsync indicates the number of overflows - THIS IS NEW IN FORMAT V2.0
#                 OverflowCorrection = OverflowCorrection + T3WRAPAROUND * nsync;
#                 GotOverflow(nsync);
#               end;    
#             end;
#             if (channel >= 1) && (channel <= 15)  % these are markers
#               true_nSync = OverflowCorrection + nsync;
#               GotMarker(true_nSync, channel);
#             end;    
#         end;
#     end;
# end
# 
# %% Read HydraHarp/TimeHarp260 T2
# function ReadHT2(Version)
#     global fid;
#     global TTResult_NumberOfRecords; % Number of TTTR Records in the File;
#     global RecNum;
# 
#     OverflowCorrection = 0;
#     T2WRAPAROUND_V1=33552000;
#     T2WRAPAROUND_V2=33554432; % = 2^25  IMPORTANT! THIS IS NEW IN FORMAT V2.0
# 
#     for i=1:TTResult_NumberOfRecords
#         RecNum = i;
#         T2Record = fread(fid, 1, 'ubit32');     % all 32 bits:
#         %   +-------------------------------+  +-------------------------------+ 
#         %   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|
#         %   +-------------------------------+  +-------------------------------+  
#         dtime = bitand(T2Record,33554431);   % the last 25 bits:
#         %   +-------------------------------+  +-------------------------------+ 
#         %   | | | | | | | |x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|
#         %   +-------------------------------+  +-------------------------------+
#         channel = bitand(bitshift(T2Record,-25),63);   % the next 6 bits:
#         %   +-------------------------------+  +-------------------------------+ 
#         %   | |x|x|x|x|x|x| | | | | | | | | |  | | | | | | | | | | | | | | | | |
#         %   +-------------------------------+  +-------------------------------+
#         special = bitand(bitshift(T2Record,-31),1);   % the last bit:
#         %   +-------------------------------+  +-------------------------------+ 
#         %   |x| | | | | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |
#         %   +-------------------------------+  +-------------------------------+
#         % the resolution in T2 mode is 1 ps  - IMPORTANT! THIS IS NEW IN FORMAT V2.0
#         timetag = OverflowCorrection + dtime;
#         if special == 0   % this means a regular photon record
#            GotPhoton(timetag, channel + 1, 0)
#         else    % this means we have a special record
#             if channel == 63  % overflow of dtime occured
#               if Version == 1
#                   OverflowCorrection = OverflowCorrection + T2WRAPAROUND_V1;
#                   GotOverflow(1);
#               else              
#                   if(dtime == 0) % if dtime is zero it is an old style single oferflow
#                     OverflowCorrection = OverflowCorrection + T2WRAPAROUND_V2;
#                     GotOverflow(1);
#                   else         % otherwise dtime indicates the number of overflows - THIS IS NEW IN FORMAT V2.0
#                     OverflowCorrection = OverflowCorrection + T2WRAPAROUND_V2 * dtime;
#                     GotOverflow(dtime);
#                   end;
#               end;
#             end;
#             if channel == 0  % Sync event
#                 GotPhoton(timetag, channel, 0);
#             end;
#             if (channel >= 1) && (channel <= 15)  % these are markers
#                 GotMarker(timetag, channel);
#             end;    
#         end;
#     end;
# end

# %% Read PicoHarp T3
# function ReadPT3
#     global fid;
#     global fpout;
#     global RecNum;
#     global TTResult_NumberOfRecords; % Number of TTTR Records in the File;
#     ofltime = 0;
#     WRAPAROUND=65536;  
# 
#     for i=1:TTResult_NumberOfRecords
#         RecNum = i;
#         T3Record = fread(fid, 1, 'ubit32');     % all 32 bits:
#     %   +-------------------------------+  +-------------------------------+ 
#     %   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|
#     %   +-------------------------------+  +-------------------------------+    
#         nsync = bitand(T3Record,65535);       % the lowest 16 bits:  
#     %   +-------------------------------+  +-------------------------------+ 
#     %   | | | | | | | | | | | | | | | | |  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|
#     %   +-------------------------------+  +-------------------------------+    
#         chan = bitand(bitshift(T3Record,-28),15);   % the upper 4 bits:
#     %   +-------------------------------+  +-------------------------------+ 
#     %   |x|x|x|x| | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |
#     %   +-------------------------------+  +-------------------------------+       
#         truensync = ofltime + nsync;
#         if (chan >= 1) && (chan <=4)
#             dtime = bitand(bitshift(T3Record,-16),4095);
#             GotPhoton(truensync, chan, dtime);  % regular count at Ch1, Rt_Ch1 - Rt_Ch4 when the router is enabled
#         else
#             if chan == 15 % special record
#                 markers = bitand(bitshift(T3Record,-16),15); % where these four bits are markers:     
#     %   +-------------------------------+  +-------------------------------+ 
#     %   | | | | | | | | | | | | |x|x|x|x|  | | | | | | | | | | | | | | | | |
#     %   +-------------------------------+  +-------------------------------+
#                 if markers == 0                           % then this is an overflow record
#                     ofltime = ofltime + WRAPAROUND;         % and we unwrap the numsync (=time tag) overflow
#                     GotOverflow(1);
#                 else                                    % if nonzero, then this is a true marker event
#                     GotMarker(truensync, markers);
#                 end;
#             else
#                 fprintf(fpout,'Err ');
#             end;
#         end;    
#     end;    
# end
# 

if __name__ == '__main__':
    #phd = PicoHarpData('cdse-singlextal-0001-mtixtl-lifetime.phd')
    
    ptu = PicoHarpPTU(r"C:\Data\dgarfield\15_07_01\test.ptu")
