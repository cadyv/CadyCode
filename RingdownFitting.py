import numpy as np
import matplotlib.pyplot as mpl
from scipy.signal import decimate
from scipy.optimize import differential_evolution
from scipy.optimize import curve_fit

def exp_floor(T, c0, c1, c_floor):
    return np.exp(c0+c1*T)+c_floor

def sumSqErr_exp_floor(parameterTuple, t_fit, amp_fit):
    val = exp_floor(t_fit, *parameterTuple)
    return np.sum((amp_fit - val) ** 2.0)

def gen_init_params_exp_floor(t_fit, amp_fit):
    parameterBounds = []
    parameterBounds.append([0.0, 10.0]) # search bounds for a
    parameterBounds.append([-1e5, 0]) # search bounds for b
    parameterBounds.append([0, 5])

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumSqErr_exp_floor, parameterBounds, args=[t_fit, amp_fit], seed=3)
    return result.x


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

def fit_single_ringdown(times, amp, f_approx, **kwargs):

    ############### Default params ##################
    decimation = kwargs.pop('decmiation', 50) # Pick this based on how oversampled you are
    harmonics = kwargs.pop('harmonics', np.arange(10)) # Increasing this makes it run a bit slower probably

    f_fit_iter = kwargs.pop('f_fit_iter', 1)
    method = kwargs.pop('method', 'logfit')
    startbuff = kwargs.pop('startbuff', 0.01) # what fraction of data to throw away at the start/end
    endbuff = kwargs.pop('endbuff', 0.01)

    minPhiSamples = kwargs.pop('minPhiSamples', 10)
    max_expected_detuning = kwargs.pop('max_expected_detuning', 20000) # Frequency detuning
    max_THD = kwargs.pop('max_THD', 10)
    start_max_THD = kwargs.pop('start_max_THD', max_THD)

    printThings = kwargs.pop('printThings', True)
    showPLTs = kwargs.pop('showPLTs', True)

    for i in np.arange(f_fit_iter):
        # Take a software Lockin to get the decay envelope and downsample the data
        LIdata = streamlinedLockIn(times, amp, f_approx, decimation = decimation, harmonics = harmonics)

        t = LIdata[1:,0]
        r = LIdata[1:,3]
        phi = LIdata[1:,4]
        inds = np.arange(len(t))
        decimated_sample_rate = 1/(t[0]-t[1])

        startBuffSamples = max(int(startbuff*len(r)), 1)
        endBuffSamples = -max(int(endbuff*len(r)), 1)

        firstphi = startBuffSamples

        df = np.diff(phi)/360 * decimated_sample_rate
        jumps = np.where(np.abs(df) > max_expected_detuning)
        jumpIndices = inds[0:-1][jumps]
        if len(jumps[0])>0:
            lastphi = np.min([len(phi)+endBuffSamples, np.min(jumpIndices[jumpIndices > firstphi]) - 1])
        else:
            lastphi = len(phi)+endBuffSamples

        # Fit and plot the phase, extract the corrected frequency

        phicoeffs = np.polyfit(t[firstphi:lastphi],phi[firstphi:lastphi], 1)
        detuning = phicoeffs[0]/360
        f_approx = f_approx + detuning
        if printThings:
            print(f'Fitted phase iter {i} using {len(phi[firstphi:lastphi])} points, freq = {f_approx:.2f}')
    sumf = f_approx

    THD = np.sqrt((LIdata[1:,5::2]**2).sum(1))/r

    startgoodTHD = np.where(THD < start_max_THD)

    firstr = np.max([startBuffSamples, startgoodTHD[0][0]])
    badTHD = np.where(THD > max_THD)
    THDIndices = inds[badTHD]
    if len(THDIndices[THDIndices > firstr])>0:
        lastr = np.min([len(r)+endBuffSamples, np.min(THDIndices[THDIndices > firstr]) - 1])
    else:
        lastr = len(r)+endBuffSamples

    if showPLTs:
        alldat = '#030301'
        fitdat = '#00D9C0'
        fitfit = '#B7AD99'
        baddat = '#FF4365'
        mpl.close('all')
        fig, ax = mpl.subplots(2,2, figsize=(8,6))
        ax[0,0].set_title('Amplitude fit')
        ax[0,0].plot(t, np.log(r), c=alldat, label='All Data')
        ax[0,0].plot(t[firstr:lastr], np.log(r[firstr:lastr]), c=fitdat, label='Fitted Data')
        ax[0,1].set_title('Phase fit')
        ax[0,1].plot(t, phi, c=alldat, label='All Data')
        ax[0,1].plot(t[firstphi:lastphi], phi[firstphi:lastphi], c=fitdat,label='Fitted Data')
        ax[0,1].plot(t, t*phicoeffs[0] + phicoeffs[1], label='Fit', c=fitfit, lw=2)
        ax[1,1].set_title('Phase jumps (cutoff for phase fit)')
        ax[1,1].hlines([max_expected_detuning, -max_expected_detuning], t[0],t[-1], color=fitfit, linestyle='dashed', label='max exp. detuning')
        ax[1,1].plot(t[1:], df, c=alldat, label='df')
        ax[1,1].plot(t[1:lastphi+1], df[:lastphi], c=fitdat)
        ax[1,1].scatter(t[1:][jumps], df[jumps], label='Detected jumps', c=baddat)
        ax[1,0].set_title('Total Harmonic Distortion (cutoff for amplitude fit)')
        ax[1,0].hlines(max_THD, t[0], t[-1], color=fitfit, linestyle='dashed', label='max THD fitted')
        ax[1,0].hlines(start_max_THD, t[0], t[-1], color=fitfit, linestyle='dashed', label='max THD fitted')
        ax[1,0].plot(t, THD, c=alldat)
        ax[1,0].scatter(t[badTHD], THD[badTHD], c=baddat)
        ax[1,0].scatter(t[0:firstr-1], THD[0:firstr-1], c=baddat)
        ax[1,0].plot(t[firstr:lastr],THD[firstr:lastr], c=fitdat)

    # Fit and plot the amplitude with either method, and extract Q
    if method == 'exp_floor':
        if printThings:
            print(f'Fitting amplitude using exponential fit with floor and {len(r[firstr:lastr])} points')
        init_params = gen_init_params_exp_floor(t[firstr:lastr], r[firstr:lastr])
        bounds = ([0, -1e5,0],[10.0, 0, 5])
        rcoeffs, pcov1 = curve_fit(exp_floor, t[firstr:lastr], r[firstr:lastr], p0=init_params, bounds = bounds)
        if showPLTs:
            ax[0,0].plot(t, np.log(exp_floor(t, *rcoeffs)), label='Fits', c=fitfit, lw=2)
        tau = -1/rcoeffs[1]
        Q = tau*sumf*np.pi

    elif method == 'logfit':
        if printThings:
            print(f'Fitting log(amplitude) with linear fit and {len(r[firstr:lastr])} points')
        rcoeffs = np.polyfit(t[firstr:lastr], np.log(r[firstr:lastr]), 1)
        ax[0,0].plot(t, t*rcoeffs[0] + rcoeffs[1], label='Fits', c=fitfit, lw=2)
        # amplitude decay time in seconds/e-folding
        tau = -1/rcoeffs[0]
        Q = tau*sumf*np.pi

    else:
        print('invalid method')
        rcoeffs = np.asarray((np.nan, np.nan))

    if showPLTs:
        ax[0,0].set_xlabel('time (s)')
        ax[0,1].set_xlabel('time (s)')
        ax[0,0].set_ylabel('log(r)')
        ax[0,1].set_ylabel('phi')
        ax[0,0].legend()
        ax[0,1].legend()
        ax[1,1].legend
        fig.tight_layout()
    return Q, sumf
