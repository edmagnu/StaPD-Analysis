from math import *
from numpy import *
from matplotlib.pyplot import *
from scipy import special, optimize

def l_lim( axlim ):
    # Loose axis limits
    axmin, axmax = axlim
    axrng = axmax - axmin
    new_min = axmin - 0.1*axrng
    new_max = axmax + 0.1*axrng
    return new_min, new_max

def nd_sort(ndarray, column):
    # Sort an ndarray (what loadtxt spits out)
    sort_indicies = ndarray[:,column].argsort()
    return ndarray[sort_indicies]

def field( a ):
    # Given a MW Attenuation (as read on the variable attenuator), return the cavity field in V/cm
    # -----
    # a : MW Attenuation as read on the variable attenuator
    return 7.58e-6 + 17.05*exp( -(a - 21.2)/8.69 )

def lin_ave(data, n):
    # Produce a running average from the nearest 2n+1 points
    # Works on 1-d arrays or columns in 2-d arrays
    # -----
    # data : array to average
    # n    : 2n+1 points to average
    dim = data.shape
    if size(dim) == 1:
        l = dim[0] - 2*n
        m = 0
        data_ave = zeros( l )
    else:
        l = dim[0] - 2*n
        m = dim[1]
        data_ave = zeros( [l,m] )
    
    for i in range(0, l):
        ave = mean( data[arange(i, i + 2*n+1)], axis=0 )
        data_ave[i] = ave
    
    return data_ave

# def CosFit(x,y,ini_guess=[0.0,0.0,0.0,0.0]):
    # # Fit dataset with a cosine function defined by : y -> p[0]+p[1]*cos(2*pi*(x-p[2])/p[3])
    # # dataset : Datset as created by the load_data function
    # # ini_guess : Vector contaning the intial set of fitting parameters, if empty the function
    # # will make a guess.
    # #from scipy import optimize
    
    # # Guesses
    # if allclose(ini_guess,[0.0,0.0,0.0,0.0]):
        # ini_guess[0] = y.mean()
        # ini_guess[1] = 0.5*(y.max()-y.min())
        # ini_guess[2] = 0.0
        # ini_guess[3] = 0.5
    
    # # Fit and error functions for optimizing.
    # fitfunc = lambda p, x: p[0]+p[1]*cos(2*pi*(x-p[2])/p[3]) # Target function
    # errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
    
    # # work the fit and pull out the diagonal covariances
    # p1, cov, infodict, bull1, bull2 = optimize.leastsq(errfunc, ini_guess[:], args=(x, y),full_output=True)
    # perr = sqrt(diag(cov))
    
    # # Get the mean square error to calculate the fit uncertainties
    # sqr_err = (y - fitfunc(p1, x) )**2
    # mean_sqr_err = sum(sqr_err)/(len(x) - len(p1))
    # p_unc = (perr * mean_sqr_err )**(0.5)
    
    # if abs(p1[3]-0.5) > 0.15:
        # print('More than 15\% on the wavelength, check the fit !')
    
    # # Correct weird fit results
    # # Amplitude is always positive
    # if p1[1] < 0 :
        # p1[1] = -p1[1]          # Flip ampltidue
        # p1[2] = p1[2] - p1[3]/2 # Offset phase by pi
    # # Phase shouldn't extend past -lambda/2 < phi < lambda/2
    # if ( p1[2] < -p1[3]/2 or p1[2] > p1[3]/2 ):
        # p1[2] = p1[2] % p1[3]  # Bring phase within 0 < phi < lambda
        # if p1[2] > p1[3]/2:    # Move phase to -lambda/2 < phi < lambda/2
            # p1[2] = p1[2] - p1[3]
        
    # return p1, p_unc

def CosFit(x,y,ini_guess=[0.0,0.0,0.0]):
    # Fit dataset with a cosine function defined by : y -> p[0]+p[1]*cos(2*pi*(x-p[2])/wl)
    # Note Fixed wavelength!
    # dataset : Datset as created by the load_data function
    # ini_guess : Vector contaning the intial set of fitting parameters, if empty the function
    # will make a guess.
    #from scipy import optimize
    
    wl = 0.5 # Fixed wavelength in fit.
    
    # Guesses
    if allclose(ini_guess,[0.0,0.0,0.0]):
        ini_guess[0] = y.mean()
        ini_guess[1] = 0.5*(y.max()-y.min())
        ini_guess[2] = 0.0
    
    # Fit and error functions for optimizing.
    fitfunc = lambda p, x: p[0]+p[1]*cos(2*pi*(x-p[2])/wl) # Target function
    errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
    
    # work the fit and pull out the diagonal covariances
    p1, cov, infodict, bull1, bull2 = optimize.leastsq(errfunc, ini_guess[:], args=(x, y),full_output=True)
    perr = sqrt(diag(cov))
    
    # Get the mean square error to calculate the fit uncertainties
    sqr_err = (y - fitfunc(p1, x) )**2
    mean_sqr_err = sum(sqr_err)/(len(x) - len(p1))
    p_unc = (perr * mean_sqr_err )**(0.5)
    
    # Correct weird fit results
    # Amplitude is always positive
    if p1[1] < 0 :
        p1[1] = -p1[1]          # Flip ampltidue
        p1[2] = p1[2] - wl/2 # Offset phase by pi
    # Phase shouldn't extend past -lambda/2 < phi < lambda/2
    if ( p1[2] < -wl/2 or p1[2] > wl/2 ):
        p1[2] = p1[2] % wl  # Bring phase within 0 < phi < lambda
        if p1[2] > wl/2:    # Move phase to -lambda/2 < phi < lambda/2
            p1[2] = p1[2] - wl
        
    return p1, p_unc
    
def mw_phase(step, freq):
    # return the phase in units of 1/wavelength
    # -----
    # step : Step of the delay translation stage
    # freq : frequency of microwaves in the cavity
    c = 299792458       # speed of light (m/s)
    wlength = c/freq    # mw wavelength (m)
    conv = 2.032095e-7  # convert steps to distance (m/step)
    x = 2*conv*step     # delay (m)
    phase = x/wlength   # delay ( 1/mw wavelength )
    
    return phase

def analyze(filename, mw_freq, n):
    # Given a data file, return delay (1/wavelengths) and normalized signal
    # -----
    # filename : file containing experiment data in the format
    # (0 iteration | 1 delay step | 2 norm | 3 norm bkgnd | 4 signal | 5 signal bkgnd)
    # mw_freq  : Microwave frequency (Hz)
    # n        : 2n+1 points in the running average of the norm data.
    
    # Breakdown the data file
    data   = loadtxt(filename)
    steps  = data[:,1]
    delay  = mw_phase(steps, mw_freq)
    norm   = data[:,2] - data[:,3]
    signal = data[:,4] - data[:,5]
    
    # Running average of the norm data, trim delay and signal accordingly
    norm   = lin_ave(norm, n)
    signal = signal[ arange(n, len(norm) + n ) ]
    normal = signal/norm
    delay  = delay[ arange(n, len(norm) + n ) ]
    
    return delay, normal

def fit_print(filename, mw_freq, n):
    # Print the fit parameters of a delay scan.
    # Updated to work with fixed wavelength CosFit()
    
    wl = 0.5 # Fixed wavelength of fit in terms of MW wavelengths
    
    delay, normal = analyze(filename, mw_freq, 1)
    fitfunc = lambda p, delay: p[0] + p[1]*cos( 2*pi*(delay - p[2]) / wl ) # Target function
    p, p_unc =  CosFit(delay,normal,[0.0,0.0,0.0]) # Fit paramters and errors
    
    str_mean = '\n\t mean       = %.2e +/- %.2e' % ( p[0], p_unc[0] )
    str_amp  = '\n\t amplitude  = %.2e +/- %.2e' % ( p[1], p_unc[1] )
    cont = abs( p[1]/p[0] )
    cont_unc = ( (1/p[0])**2 * p_unc[1]**2 + (p[1]/p[0]**2)**2 * p_unc[0]**2 )**(0.5)
    str_cont = '\n\t contrast   = %.3f +/- %.3f' % ( cont, cont_unc )
    str_wln  = '\n\t wavelength = %.3f +/- %.3f' % ( wl,   0.0 ) # Updated to use fixed wavelength CosFit()
    str_phi  = '\n\t phase      = %.3f +/- %.3f' % ( p[2], p_unc[2] )
    str_fit  = str_mean + str_amp + str_cont + str_wln + str_phi
    
    print('\n' + filename + ' ' + 20*'-' + str_fit)
    
    # plot(delay, normal, label = filename)
    # plot(delay, fitfunc(p, delay), linewidth = 3)
    
    return

def dyescan(filename, n):
    data = loadtxt(filename)
    data = lin_ave(data, n)
    freq      = data[:,1]
    norm      = data[:,2]
    norm_bk   = data[:,3]
    signal    = data[:,4]
    signal_bk = data[:,5]
    
    nsig      = (signal - signal_bk)/(norm - norm_bk)
    
    return freq, nsig