from NuRadioReco.utilities import units
from NuRadioReco.modules.io.coreas import coreas
from radiotools import coordinatesystems as coordinatesystems
import pandas as pd
import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import logging

sns.set()

#to be changed
sys.path.append("/home/pysia/Pulpit/RNO-G/cr-pulse-interpolator/")
import interpolation_fourier as interpF
import signal_interpolation_fourier as sigF
import demo_helper

eFieldCgsToSI = 2.99792458e10 * units.micro * units.volt / units.meter


def get_freq_axis(signal, sampling_period=0.2e-9):
    return (1.0e-6 * np.fft.rfftfreq(signal.shape[0], d=sampling_period)) # MHz

 
def do_filter_signal_bandpass(signal, low_cutoff, high_cutoff):
    
    freqs = get_freq_axis(signal)
    filter_indices = np.where( (freqs < low_cutoff) | (freqs > high_cutoff))
    spectrum = np.fft.rfft(signal)
    spectrum[filter_indices] *= 0.0
    signal_filtered = np.fft.irfft(spectrum)

    return signal_filtered


class testInterpolation():

    def __init__(self):
        """
        __init__ method
        """        
        self.__input_files = None
        self.__output_folder = None    
        self.logger = logging.getLogger('NuRadioReco.energyInterpolationCoREAS')
    def begin(self, input_files, output_folder = "", testSignal = True, testEnergy = False, testRingsNumber = [0], save_output = 0, low_cutoff_MHz = 30, high_cutoff_MHz = 500, sampling_period = 0.2e-9):
        """
        begin method

        initialize energyInterpolationCoREASStation module

        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        output_folder: output folder (default = "")
            folder where the plots are saved
        testSignal: flag (default = True)
            perform electric signal test
        testEnergy: flag (default = False)
            perform energy test
        testRingsNumber: list (default = [0])
            indices of the rings to be used for the test !! WARNING!! ring numbers are not ordered in shower plane, e.g. 0 ring is placed in the middle of the ring
            range from the 0 to 30
        save_output: flag (default = 0)
            save or discard the output
        low_cutoff_MHz: float (default = 30)
            lower frequency for the bandpass filter in interpolation
        high_cutoff_MHz: float (default = 500)
            higher frequency for the bandpass filter in interpolation
        sampling_period: float (default 0.2e-9)
            sampling period of the signal
        """
        self.__input_files = input_files
        self.__output_folder = output_folder    
        self.testSignal = testSignal
        self.testEnergy = testEnergy
        self.save_output = save_output
        self.testRingsNumber = testRingsNumber
        self.low_cutoff_MHz = low_cutoff_MHz
        self.high_cutoff_MHz = high_cutoff_MHz
        self.sampling_period = sampling_period

    def run(self):
        """
         Electric field interpolation test and/or energy iterpolation test - input simulation file contains 240 observe positions creating star shape with 8 arms. 
         A given number of N (testRingsNumber) rings in taken, i.e. N*8 observe positions, for the test and remaining observe positions are used for the interpolation.

        """
        for filePath in self.__input_files: 
           
            try:
                coreasSimulationData = h5py.File(filePath, "r")
            except:
                self.logger.warning("could not open >> {0} << \n".format(filePath))
                continue

            fileName = filePath[filePath.find("SIM"): filePath.find(".hdf5")]            
            observersData = coreasSimulationData["CoREAS"]["observers"]
            observersDataNames = coreasSimulationData["CoREAS"]["observers"].keys()
     
            positionsNuCoord = []
            signalsNuCoord = []
            energy = []

            for on in observersDataNames:

                # transformation of the observe positions from the CoREAS coordinate system to NuRadioReco & cgs -> SI  
                positionCoREAS = observersData[on].attrs['position']    
                positionsNuCoord.append(np.array([-positionCoREAS[1], positionCoREAS[0], 0]) * units.cm)  


                # transformation of the electic field from the CoREAS coordinate system to NuRadioReco & cgs -> SI  
                # electricFieldData[:,i], i = 0, 1, 2, 3 -> respectively: time, Ex, Ey, Ez
                electricFieldData = observersData[on][()]
                signalsNuCoord.append(np.array([electricFieldData[:,0]*units.second, -electricFieldData[:,2]*eFieldCgsToSI, electricFieldData[:,1]*eFieldCgsToSI, electricFieldData[:,3]*eFieldCgsToSI]).T) 

                electricFieldStrengthSquaredSum = 0
                for e in electricFieldData:
                    electricFieldStrengthSquaredSum+=(np.sum(np.power(np.array(e[1:]),2)))
                energy.append(electricFieldStrengthSquaredSum)
                        
            #convert to np arrays
            positionsNuCoord = np.array(positionsNuCoord)
            signalsNuCoord = np.array(signalsNuCoord)
            energy = np.array(energy) 
            
            #radiotools package for the coordinate system transformations
            zenith, azimuth, magnetic_field_vector = coreas.get_angles(coreasSimulationData)
            cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector)

            #transform observe positions to shower plane
            positionsGeographic = cs.transform_from_magnetic_to_geographic(positionsNuCoord.T) 
            positionsShowerPlane = cs.transform_to_vxB_vxvxB(positionsGeographic).T 

            #transform eField to on-sky coordinate system
            electricFieldOnSky = []
            for signal in signalsNuCoord:
                signalGeographic = cs.transform_from_magnetic_to_geographic(signal[:,1:].T) 
                signalOnSky = cs.transform_from_ground_to_onsky(signalGeographic)
                electricFieldOnSky.append(np.insert(signalOnSky.T, 0, signal[:,0], axis = 1)) #add time column to signalOnSky numpy array

            electricFieldOnSky = np.array(electricFieldOnSky)

            #extract observe positions in the shower plane 
            coordVB = positionsShowerPlane[:,0]
            coordVVB = positionsShowerPlane[:,1]
            starIndices = np.arange(0, 240)

            #visualize the observe positions in shower plane with numbering
            plt.scatter(coordVB, coordVVB, s = 3)
            for i in range(len(starIndices)):
                plt.annotate(str(starIndices[i]), (coordVB[i], coordVVB[i]), textcoords="offset points", xytext=(0.5,0.5), ha='left', fontsize =8)
            plt.xlabel('VxB [m]')
            plt.ylabel('VxVxB [m]')
            plt.gca().set_aspect('equal')
            plt.show()
            plt.clf()

            baseIndices = np.arange(0,8)

            # indices of the observe position for the test
            testIndices = np.concatenate([baseIndices + (N * 8) for N in self.testRingsNumber])
            print()
            # indices of the observe positions for the interpolation
            interpIndices =  np.setdiff1d(starIndices, testIndices) 

            # select positions and electric field of the observe positions used for the interpolation
            interpCoordVB = coordVB[interpIndices]
            interpCoordVVB = coordVVB[interpIndices]
            interpElectricFieldOnSky = electricFieldOnSky[interpIndices]
            interpEnergy = energy[interpIndices]
            #random indices for the test 
            randomTestIndices = np.random.choice(testIndices, size = 1, replace = True)
            randomTestCoordVB = coordVB[randomTestIndices]
            randomTestCoordVVB = coordVVB[randomTestIndices]
            randomElectricFieldOnSky = electricFieldOnSky[randomTestIndices]
            randomTestEnergy = energy[randomTestIndices]

            #interpolation 
            if self.testSignal:
                print("\n\n  ==============================================================================")
                print(" | Signal interpolation test in file {0} ".format(fileName))   
                print("  ==============================================================================\n\n")   
                signal_interpolator = sigF.interp2d_signal(interpCoordVB, interpCoordVVB, interpElectricFieldOnSky[:,:,1:], lowfreq = self.low_cutoff_MHz, highfreq = self.high_cutoff_MHz,  sampling_period= self.sampling_period, verbose=False, phase_method='phasor')
                for n,(x,y,i) in enumerate(zip(randomTestCoordVB, randomTestCoordVVB,randomTestIndices)):
                    
        
                    print("test position at (VxB, VxVxB): ({0},{1})".format(x,y))
                    """
                    filtered_onsky_efield = do_filter_signal_bandpass(electricFieldOnSky[i, :, 1], self.low_cutoff_MHz, self.high_cutoff_MHz)
                    plt.plot(filtered_onsky_efield, label = "sim filtered")
                    plt.plot(signal_interpolator(x,y)[:,0], label = "interp")
                    plt.xlabel("time [s]")
                    plt.ylabel("$E_{R} [V/m]$")
                    plt.legend() 
                    if self.save_output == 1:
                        plt.savefig("{0}Er_time_atObservePosition{1}.pdf".format(self.__output_folder, i))
                    plt.show()
                    plt.clf()
                    cc = demo_helper.get_crosscorrelation(signal_interpolator(x,y)[:,0], filtered_onsky_efield)
                    print("r component, cross correleation in time domain: ", cc)
 
                    simSignalFftPower = np.abs(np.fft.rfft(filtered_onsky_efield))**2
                    interpSignalFftPower = np.abs(np.fft.rfft(signal_interpolator(x,y)[:,0]))**2
                    freq = get_freq_axis(filtered_onsky_efield, sampling_period = self.sampling_period)
                    plt.plot( freq, simSignalFftPower, label = "sim")
                    plt.plot( freq, interpSignalFftPower, label = "interp")
                    plt.xlabel("frequency [MHz]")
                    plt.ylabel("$|E_{R}|^{2} [V^{2}/m^{2}]$")
                    plt.tight_layout()
                    plt.legend()
                    if self.save_output == 1:
                        plt.savefig("{0}Er_frequency_atObservePosition{1}.pdf".format(self.__output_folder,i))
                    plt.show()
                    plt.clf()
                    cc = demo_helper.get_crosscorrelation(interpSignalFftPower, simSignalFftPower)
                    print("r component, cross correleation in frequency domain: ", cc)
                    """
                    filtered_onsky_efield = do_filter_signal_bandpass(electricFieldOnSky[i, :, 2], self.low_cutoff_MHz, self.high_cutoff_MHz)
                    plt.plot(filtered_onsky_efield, label = "sim filtered")
                    plt.plot(signal_interpolator(x,y)[:,1], label = "interp")
                    plt.xlabel("time [s]")
                    plt.ylabel("$E_{\theta} [V/m]$")
                    plt.legend()
                    if self.save_output == 1:
                        plt.savefig("{0}Etheta_time_atObservePosition{1}.pdf".format(self.__output_folder,i))
                    plt.show()
                    plt.clf()
                    cc = demo_helper.get_crosscorrelation(signal_interpolator(x,y)[:,1], filtered_onsky_efield)
                    dane = [str(i), str(energy[i]), str(cc[1]),"\n" ]
                    print(dane)
                    print("theta component, cross correleation in time domain: ", cc)
                    with open("theta.txt", 'a') as plik:
                        plik.write(','.join(dane))

                    simSignalFftPower = np.abs(np.fft.rfft(filtered_onsky_efield))**2
                    interpSignalFftPower = np.abs(np.fft.rfft(signal_interpolator(x,y)[:,1]))**2
                    freq = get_freq_axis(filtered_onsky_efield, sampling_period = self.sampling_period)
                    plt.plot( freq, simSignalFftPower, label = "sim")
                    plt.plot( freq, interpSignalFftPower, label = "interp")
                    plt.xlabel("frequency [MHz]")
                    plt.ylabel("$|E_{\theta}|^{2} [V^{2}/m^{2}]$")
                    plt.tight_layout()
                    plt.xlim(0,1000)
                    plt.legend()
                    if self.save_output == 1:
                        plt.savefig("{0}Etheta_frequency_atObservePosition{1}.pdf".format(self.__output_folder,i))
                    
                    plt.show()
                    plt.clf()
                    plt.clf()
                    cc = demo_helper.get_crosscorrelation(interpSignalFftPower, simSignalFftPower)
                    print("theta component, cross correleation in frequency domain: ", cc)

                    filtered_onsky_efield = do_filter_signal_bandpass(electricFieldOnSky[i, :, 3], self.low_cutoff_MHz, self.high_cutoff_MHz)
                    plt.plot(filtered_onsky_efield, label = "sim filtered")
                    plt.plot(signal_interpolator(x,y)[:,2], label = "interp")
                    plt.xlabel("time [s]")
                    plt.ylabel("$E_{\phi} [V/m]$")
                    plt.legend()
                    if self.save_output == 1:
                        plt.savefig("{0}Ephi_time_atObservePosition{1}.pdf".format(self.__output_folder,i))
                    plt.show()
                    plt.clf()
                    cc = demo_helper.get_crosscorrelation(signal_interpolator(x,y)[:,2], filtered_onsky_efield)
                    print("theta component, cross correleation in time domain: ", cc)
                    dane = [str(i), str(energy[i]), str(cc[1]),"\n" ]
                    print(dane)
                    with open("phi.txt", 'a') as plik:
                        plik.write(','.join(dane))

                    simSignalFftPower = np.abs(np.fft.rfft(filtered_onsky_efield))**2
                    interpSignalFftPower = np.abs(np.fft.rfft(signal_interpolator(x,y)[:,2]))**2
                    freq = get_freq_axis(filtered_onsky_efield, sampling_period = self.sampling_period)
                    plt.plot( freq, simSignalFftPower, label = "sim")
                    plt.plot( freq, interpSignalFftPower, label = "interp")
                    plt.xlabel("frequency [MHz]")
                    plt.ylabel("$|E_{\phi}|^{2} [V^{2}/m^{2}]$")
                    plt.tight_layout()
                    plt.legend()
                    plt.xlim(0,1000)    
                    if self.save_output == 1:
                        plt.savefig("{0}Ephi_frequency_atObservePosition{1}.pdf".format(self.__output_folder,i))

                    plt.show()
                    plt.clf()
                    cc = demo_helper.get_crosscorrelation(interpSignalFftPower, simSignalFftPower)
                    print("phi component, cross correleation in frequency domain: ", cc)


            if self.testEnergy:
                print("\n\n  ==============================================================================")
                print(" | Energy interpolation test in file {0} ".format(fileName))   
                print("  ==============================================================================\n\n")   
                fourierInterpolator = interpF.interp2d_fourier(interpCoordVB, interpCoordVVB, interpEnergy)
                interpPointsVxB = sorted(coordVB)  # now exact poositions are taken from star shape for the interpolation
                interpPointsVxVxB =  sorted(coordVVB) 
                X, Y = np.meshgrid(interpPointsVxB, interpPointsVxVxB)
                Z = fourierInterpolator(X,Y)
                for cVB, cVVB, iT in zip(randomTestCoordVB, randomTestCoordVVB, randomTestIndices):
                    print("test position at (VxB, VxVxB): ({0},{1})".format(cVB,cVVB))
                    minDiffVB =  min(interpPointsVxB, key = lambda x: abs(x-cVB))    
                    minDiffVVB = min(interpPointsVxVxB, key = lambda x: abs(x-cVVB))
                    iVB = np.where(interpPointsVxB == minDiffVB)
                    iVVB = np.where(interpPointsVxVxB == minDiffVVB)
                    interpE = Z[iVVB,iVB][0][0]
                    E = energy[iT]

                    print("Energy interpolation -  relative error: ", round(abs(interpE-E)/E*100,3), "%")

    def end(self):
        pass
