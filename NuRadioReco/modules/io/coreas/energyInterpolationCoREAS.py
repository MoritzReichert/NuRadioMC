from NuRadioReco.utilities import units
from NuRadioReco.modules.io.coreas import coreas
from radiotools import coordinatesystems as cstrafo
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
sys.path.append("/home/pysia/Pulpit/RNOG/cr-pulse-interpolator/")
import interpolation_fourier as interpF

class energyInterpolationCoREAS():

    def __init__(self):
        """
        __init__ method
        """        
        self.logger = logging.getLogger('NuRadioReco.energyInterpolationCoREAS')

    def begin(self, input_files, output_folder = "", profileVxB = None, profileVxVxB = None):
        """
        begin method

        initialize energyInterpolationCoREASStation module

        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        output_folder: output folder (default = "")
            folder where the plots are saved
        profileVxB: list of points on the VxB axis (default = None)
            list of VxB points for which the energy profiles are created
        profileVxVxB: list of points on the VxVxB axis (default = None)
            list of VxVxB points for which the energy profiles are created
        """
        self.__input_files = input_files
        self.__output_folder = output_folder    
        self.profileVxB = profileVxB
        self.profileVxVxB = profileVxVxB

    def run(self):
        for filePath in self.__input_files: 
           
            fileName = filePath[filePath.find("SIM"): filePath.find(".hdf5")]
            coreasSimulationData = h5py.File(filePath, "r")
            logger.warning("could not open {0} file.".format(fileName))
            
            observersData = coreasSimulationData["CoREAS"]["observers"]
            observersDataNames = coreasSimulationData["CoREAS"]["observers"].keys()
     
            positionsNuCoord = []
            energy = []

            for on in observersDataNames:
                electricFieldData = observersData[on][()]

                electricFieldStrengthSquaredSum = 0
                for e in electricFieldData:
                    electricFieldStrengthSquaredSum+=(np.sum(np.power(np.array(e[1:]),2)))
                energy.append(electricFieldStrengthSquaredSum)
                
                positionCoREAS = observersData[on].attrs['position']
                positionsNuCoord.append(np.array([-positionCoREAS[1], positionCoREAS[0], 0]) * units.cm) 
            
            positionsNuCoord = np.array(positionsNuCoord)
            zenith, azimuth, magnetic_field_vector = coreas.get_angles(coreasSimulationData)
            cs = cstrafo.cstrafo(zenith, azimuth, magnetic_field_vector)
            positions_vBvvB = cs.transform_from_magnetic_to_geographic(positionsNuCoord.T) 
            positions_vBvvB = cs.transform_to_vxB_vxvxB(positions_vBvvB).T 

            coordVB = positions_vBvvB[:,0]
            coordVVB = positions_vBvvB[:,1]
            energy = np.array(energy)

            fourierInterpolator = interpF.interp2d_fourier(coordVB, coordVVB, energy)
            intPointsVxB = np.linspace(min(coordVB), max(coordVB), 1000)
            intPointsVxVxB = np.linspace(min(coordVVB), max(coordVVB), 1000)
            X, Y = np.meshgrid(intPointsVxB, intPointsVxVxB)
            Z = fourierInterpolator(X, Y)
            
            plt.imshow(Z, cmap='turbo', extent = [min(intPointsVxB), max(intPointsVxB), min(intPointsVxVxB), max(intPointsVxVxB)])
            colorbar = plt.colorbar()
            colorbar.set_label(r'$\mathrm{\sum_{t} E^{2}}$')
            plt.xlabel('VxB [m]')
            plt.ylabel('VxVxB [m]')
            plt.tight_layout()
            plt.savefig("{0}footprint_from_file_{1}.pdf".format(self.__output_folder, fileName))
            plt.clf()

            if (self.profileVxB != None):
                for coord in self.profileVxB:
                    c =  min(intPointsVxB, key = lambda x: abs(x-coord))
                    ind = list(intPointsVxB).index(c)
                    plt.plot(intPointsVxVxB ,Z[:,ind])
                    plt.xlabel('VxB [m]')
                    plt.ylabel(r'$\mathrm{\sum_{t} E^{2}  [V^{2}/m^{2}]}$')
                    plt.tight_layout()
                    plt.savefig("{0}profile_VxB_at_{1}_m_from_file_{2}.pdf".format(self.__output_folder, round(c,3), fileName))
                    plt.clf()

            if (self.profileVxVxB != None):
                for coord in self.profileVxVxB:
                    c =  min(intPointsVxVxB, key = lambda x: abs(x-coord))
                    ind = list(intPointsVxVxB).index(c)
                    plt.plot(intPointsVxB ,Z[ind,:])
                    plt.xlabel('VxVxB [m]')
                    plt.ylabel(r'$\mathrm{\sum_{t} E^{2}  [V^{2}/m^{2}]}$')
                    plt.tight_layout()
                    plt.savefig("{0}profile_VxVxB_at_{1}_m_from_file_{2}.pdf".format(self.__output_folder, round(c,3), fileName))
                    plt.clf()
    def end(self):
        pass


       
    