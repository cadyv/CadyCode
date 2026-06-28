import numpy as np
import matplotlib.pyplot as mpl
from scipy.signal import decimate

def streamlinedLockIn(times, signals, nominalFreq, **kwargs):
    ###########################################################################
    # Speed up math relative to streamlinedLockIn_old() by skipping math
    # on zeroth harmonic (DC level).
    # Much simplified version of lockin, and hopefully faster?
    # With respect to lockIn, change the following:
    #   -arbitrary number of signals, passed as an array:
    #       i.e. signals[:, 0] is first signal, signals[:, 1] is second...
    #   -no distinction between a 'carrier' and a 'signal'
    #   -doesn't fit to a carrier signal, just generates a sine at nominalFreq
    #   -hence, nominalFreq is promoted to a required argument
    #   -can lock in at multiple harmonics in one function call
    #   -doesn't perform the convolution step, as that is possibly slow,
    #       and the decimation step is sufficient to filter the timestreams
    #   -doesn't bother constraining itself to an integer number of periods
    #       of the signal
    ###########################################################################
    # provide a list of harmonics to lock in at
    harmonics = kwargs.pop('harmonics', [0, 1, 2, 3])
    # amount to decimate the output timestreams
    decimation = kwargs.pop('decimation', 25)
    # number of signals in the input data
    XY = kwargs.pop('XY', False)
    if len(np.shape(signals)) == 1:
        num_sigs = 1
        signals = np.reshape(signals, (signals.size, 1))
    elif len(np.shape(signals)) == 2:
        num_sigs = np.shape(signals)[1]
    # number of harmonics
    num_harmonics = len(harmonics)
    # loop counter
    counter = 0
    # main loop over harmonics
    for harmonic in harmonics:
        if harmonic == 0:
            # no need to perform a multiply step for the zeroth harmonic,
            # as the X quadrature is just multiplying by a constant zero
            # and the Y quadrature is a constant one. Phase is always 90
            # degrees for this set of definitions, no need to arctan.
            r = np.abs(decimate(signals, decimation, axis=0, ftype='fir'))
            phi = 90*np.sign(r)
            x = r*np.cos(phi)
            y = r*np.sin(phi)
        else:
            # Generate the quadrature pair:
            # amplitude 1V, frequency nominalFreq*harmonic,
            # phases 0 and 90 degrees, offset is 0V
            XX = returnSineOffset(times, 1, harmonic*nominalFreq, 0, 0)
            YY = returnSineOffset(times, 1, harmonic*nominalFreq, 90, 0)
            # unsmoothed multiplied signals
            xx = np.multiply(signals.T, XX).T
            yy = np.multiply(signals.T, YY).T
            # smoothed signals
            x = decimate(xx, decimation, axis=0, ftype='fir')
            y = decimate(yy, decimation, axis=0, ftype='fir')
            # polar coordinates
            r = 2*np.sqrt(x**2 + y**2) # check that this is where we want this
            phi = 180/np.pi*np.arctan2(y, x)
        # first time through the loop, so preallocate empty array to write to
        # and put the decimated time vector in it
        if counter == 0:
            data = np.zeros((np.shape(r)[0], 1+2*num_harmonics*num_sigs))
            data[:, 0] = times[::decimation]
        if XY:
            data[:, 1+2*counter::num_sigs*num_harmonics] = x
            data[:, 2+2*counter::num_sigs*num_harmonics] = y
        else:
            data[:, 1+2*counter::num_sigs*num_harmonics] = r
            data[:, 2+2*counter::num_sigs*num_harmonics] = phi
        counter = counter + 1
    ###########################################################################
    # DATA SORTING CONVENTION:
    # column 0: times, columns 1-2: r, phi data for (signal a, harmonic 1),
    # columns 3-4: r, phi data for (signal a, harmonic 2)....then signal b data
    # and so on.
    return data


def returnSineOffset(times, amplitude, frequency, phase, offset):
    # amplitude in your favorite units, frequency in Hz, phase in degrees
    return offset + amplitude*np.sin(2*np.pi*frequency*times+phase*np.pi/180)
