import scipy.io.wavfile
import python_speech_features
import numpy
import os

def waveform_to_mfcc(filename, 
                     save=None, 
                     binary=True, 
                     verbose=False, 
                     visualise=False, 
                     winlen=0.025, 
                     winstep=0.01, 
                     numcep=13, 
                     nfilt=26, 
                     nfft=512, 
                     lowfreq=0, 
                     highfreq=None, 
                     preemph=0.97, 
                     ceplifter=22, 
                     appendEnergy=True, 
                     winfunc=lambda x:numpy.ones((x,)), 
                     deltas=None):
    '''
    DESCRIPTION:
    Waveform takes a .wav format audio file and converts it to mel-frequency cepstral coefficients.
    
    PARAMETERS:
    filename - The filepath for the audio file - STRING
    save - If given a string path to a directory (remember trailing slash), the MFCCs are saved to that directory with the same name as the .wav file. Defaults to None which does not save the MFCCs. - None/STRING
    binary - If True, saves the MFCCs as a Numpy array with file extension .npy. If false, saves as .txt file. Defaults to True. - BOOLEAN
    verbose - If True, prints out various information about the wavefile and MFCCs. Defaults to False. - BOOLEAN
    visualise - If True, draws visuals that help to visualise the data. Defaults to False. - BOOLEAN
    
    PYTHON_SPEECH_FEATURES.MFCC PARAMETERS (info from docs at https://python-speech-features.readthedocs.io/en/latest/)
    Parameters:	
    signal – the audio signal from which to compute features. Should be an N*1 array
    samplerate – the samplerate of the signal we are working with.
    winlen – the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    winstep – the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    numcep – the number of cepstrum to return, default 13
    nfilt – the number of filters in the filterbank, default 26.
    nfft – the FFT size. Default is 512.
    lowfreq – lowest band edge of mel filters. In Hz, default is 0.
    highfreq – highest band edge of mel filters. In Hz, default is samplerate/2
    preemph – apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    ceplifter – apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    appendEnergy – if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    winfunc – the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    Returns:	
    A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    
    deltas - Determines whether delta or deltadelta coefficients are added to MFCC features. Default None, can be "delta" or "deltadelta" - None/STRING
    
    RETURNS:
    MFCCs in the form of a Numpy array.
    '''
    
    # Load the audio file.
    
    sr, wav = scipy.io.wavfile.read(filename)
    dur = len(wav)/sr
    nSamples = wav.shape[0]
    
    if verbose:
        print(filename+" information:\n", 
              "Sampling rate = "+str(sr)+"Hz",
              "Number of samples = "+str(nSamples), 
              "Duration = "+str(round(dur,2))+"s", 
              "Numpy array shape = "+str(wav.shape)+"\n", 
              sep="\n")
        
    # Convert the audio file to MFCCs.
    
    mfcc = python_speech_features.base.mfcc(wav, 
                                     samplerate=sr, 
                                     winlen=winlen, 
                                     winstep=winstep, 
                                     numcep=numcep, 
                                     nfilt=nfilt, 
                                     nfft=nfft, 
                                     lowfreq=lowfreq, 
                                     highfreq=highfreq, 
                                     preemph=preemph, 
                                     ceplifter=ceplifter, 
                                     appendEnergy=appendEnergy, 
                                     winfunc=winfunc)
    if verbose:
        print("MFCC information:\n", 
              "Number of windows (accounting for overlap) = "+str(dur/(winlen-(2*winstep))//2), # TODO check this formula
              "MFCC shape (no deltas) = "+str(mfcc.shape)+"\n", 
              sep="\n")
       
    # Calculate delta coefficients
    
    if deltas == "delta":
        d = python_speech_features.delta(mfcc, 2)
        mfcc = numpy.hstack((mfcc, d))
    elif deltas == "deltadelta":
        d = python_speech_features.delta(mfcc, 2)
        dd = python_speech_features.delta(d, 2)
        mfcc = numpy.hstack((mfcc, d, dd))
            
    if verbose and deltas!=None:
        print("MFCC information after adding delta coefficients:\n", 
              "MFCC shape (w/ deltas) = "+str(mfcc.shape), 
              "Final number of features = "+str(mfcc.shape[1])+"\n",
              sep="\n")
       
    # Save MFCC
    
    if save != None:
        if not binary:
            fname = save+os.path.basename(filename)[:-4]+".txt"
            numpy.savetxt(fname, mfcc, delimiter="\t")
            if verbose:
                print("Saving file to", fname)
        else:
            fname = save+os.path.basename(filename)[:-4]+".npy"
            numpy.save(fname, mfcc)
            if verbose:
                print("Saving file to", fname)
        
    return mfcc