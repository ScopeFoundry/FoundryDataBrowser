"""
N. Borys

Small script that uses the PicoHarpData class written by Ed Barnard to histogram
time differences between channel 0 and channel 1 events from a PicoHarp PT2 data
file

2013-11-20 Creation.
2014-01-02 Added named constants for the number of bins in the histogram.
2014-01-02 Added named constants for the number of bins in the histogram.
"""

from PicoharpPTU import PicoHarpPTU
import numpy as np
import pylab as pl
import time

filename = r"C:\data\Angel\171017\lifetimes\Er-enhanced-t2-100s-6mW-1.pt2"  #Don't forget to used double forward backslashes in the path name!
binCount = 1000
save_data = True    # Set this to True to export the data as a text file

#Use Ed's software to load the file
PHD = PicoHarpPTU(filename, True)

# Find all of the record indices of channel 0 events.  These mark the laser sync signal.
print('Finding channel 0 events')
c0events = np.where(PHD.t2_channels == 0)[0]

if (c0events.shape[0] <= 0):
	print("No events found on channel 0.  Cannot process file.")
	assert(c0events.shape[0] > 0)
else:
	print(str.format('Found {0:d} channel 0 events', c0events.shape[0]))


#Compute the time differences between the channel 0 events.  Should be a regular interval.
print('Calculating time difference between channel 1 events')
c0diffs = PHD.t2_times[c0events[1:]] - PHD.t2_times[c0events[:-1]]
print('done')

#Construct an empty histogram in which to bin the events.
print('Construct an empty histogram in which to bin the events.')
timeSpan = np.average(c0diffs)
hist,bins = np.histogram(np.empty(0, np.int64), bins=binCount, range=[ 0, timeSpan ] )

#For each channel 0 event, histogram the number of channel 1 events with respect to the delay after
#the channel 0 event.
for i, (start,stop) in enumerate(np.nditer([c0events[:-1],c0events[1:]])):
    if (i % (c0events.shape[0]/1000)) == 0:
	    print("%1.2f %%" % (100.*i/c0events.shape[0]))
    # Get the events for this excitation pulse
    c1events = np.where(PHD.t2_channels[start:stop] == 1)[0] + start

    # Histogram the times of the fluorescence events with respect to the excitation pulse
    hist = hist + (
      np.histogram(PHD.t2_times[c1events]-PHD.t2_times[start], bins=binCount, range=(0, timeSpan) )
      )[0]


# Save the histogram
if save_data:
	save_fname = filename+"_hist.txt"
	np.savetxt(save_fname,
	           np.transpose(np.array([bins[:-1]*4*10**-6, hist])),
			   delimiter='\t')
	print("data saved to", save_fname)

#Make a plot
fig1 = pl.figure(1)
fig1.clf()
ax1 = fig1.add_subplot(111)
ax1.plot(bins[:-1]*4*10**-6, hist, color='b')  #Note that the time units are set here.  The PH time unit is 4 ps
                                      #Then 10^-6 to convert to microseconds.

# Plot before time
ax1.plot((bins[:-1]-bins[-2])*4*10**-6, hist, color='b', alpha=0.5)

ax1.set_xlabel("dt (microseconds)")
ax1.set_ylabel("count")

fig1.show()
fig1.canvas.draw()

pl.show()
