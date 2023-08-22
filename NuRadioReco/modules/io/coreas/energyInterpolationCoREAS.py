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
        self.__input_files = None
        self.__output_folder = None    
        self.logger = logging.getLogger('NuRadioReco.energyInterpolationCoREAS')

    def begin(self, input_files, output_folder = "", profileVxB = False, profileVxVxB = False, save_output = 0, n_grid_points = 1000):
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
        save_output: flag (default = 0)
            save or discard the output
        n_grid_points: number of the grid points (default = 1000)
            number of the grid points for VxB and VxVxB coordinates, between their minimum and maximum values
        """
        self.__input_files = input_files
        self.__output_folder = output_folder    
        self.profileVxB = profileVxB
        self.profileVxVxB = profileVxVxB
        self.save_output = save_output
        self.n_grid_points = n_grid_points

    def run(self):
        """
        CoREAS observe positions are transformed to the shower coordiante system.
        Sum of the squares of electric field strength is interpolated with the interpolation_fourier module.
        run returns the inetrpolated values as a two-dimensional numpy.ndarray.

        For the profile plots, the closest interpolated value to the given input is used."
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
            intPointsVxB = np.linspace(min(coordVB), max(coordVB), self.n_grid_points)
            intPointsVxVxB = np.linspace(min(coordVVB), max(coordVVB), self.n_grid_points)
            X, Y = np.meshgrid(intPointsVxB, intPointsVxVxB)
            Z = fourierInterpolator(X, Y)
    
            plt.imshow(Z, cmap='turbo', extent = [min(intPointsVxB), max(intPointsVxB), min(intPointsVxVxB), max(intPointsVxVxB)])
            colorbar = plt.colorbar()
            colorbar.set_label(r'$\mathrm{\sum_{t} E^{2}}$')
            plt.xlabel('VxB [m]')
            plt.ylabel('VxVxB [m]')
            plt.tight_layout()
            if (self.save_output == 1): 
                plt.savefig("{0}footprint_from_file_{1}.pdf".format(self.__output_folder, fileName))
            plt.show()
            plt.clf()

            dataPointsVxB = []
            dataPointsVxVxB = []

            if (self.profileVxB != False):
                print("\n\n  ==============================================================================")
                print(" | Insert VxB cooridantes for the lateral distribution or press >> q << to quit |")   
                print("  ==============================================================================")               
                while True:
                    vxb = input()
                    if vxb == "q" or vxb == "Q":
                        break
                    else:
                        try:
                            vxb = float(vxb)
                        except:
                            print("not a number, continue providing input or press >> q << to quit")
                            continue
                        dataPointsVxB.append(vxb)
                self.profileVxB = dataPointsVxB

            if (self.profileVxVxB != False):
                print("\n\n  =================================================================================")
                print(" | Insert VxVxB cooridantes for the lateral distribution or press >> q << to quit |")   
                print("  =================================================================================")               
                while True:
                    vxvxb = input()
                    if vxvxb == "q" or vxvxb == "Q":
                        break
                    else:
                        try:
                            vxvxb = float(vxvxb)
                        except:
                            print("not a number, continue providing input or press >> q << to quit")
                            continue
                        dataPointsVxVxB.append(vxvxb)
                self.profileVxVxB = dataPointsVxVxB

            if (self.profileVxB != False):
                for coord in self.profileVxB:
                    c =  min(intPointsVxB, key = lambda x: abs(x-coord))
                    ind = list(intPointsVxB).index(c)
                    plt.plot(intPointsVxVxB ,Z[:,ind])
                    plt.xlabel('VxB [m]')
                    plt.ylabel(r'$\mathrm{\sum_{t} E^{2}  [V^{2}/m^{2}]}$')
                    plt.tight_layout()
                    if (self.save_output == 1): 
                        plt.savefig("{0}profile_VxB_coord_at_{1}_m_from_file_{2}.pdf".format(self.__output_folder, round(c,3), fileName))
                    plt.show()
                    plt.clf()

            if (self.profileVxVxB != False):
                for coord in self.profileVxVxB:
                    c =  min(intPointsVxVxB, key = lambda x: abs(x-coord))
                    ind = list(intPointsVxVxB).index(c)
                    plt.plot(intPointsVxB ,Z[ind,:])
                    plt.xlabel('VxVxB [m]')
                    plt.ylabel(r'$\mathrm{\sum_{t} E^{2}  [V^{2}/m^{2}]}$')
                    plt.tight_layout()
                    if (self.save_output == 1): 
                        plt.savefig("{0}profile_VxVxB_coord_at_{1}_m_from_file_{2}.pdf".format(self.__output_folder, round(c,3), fileName))
                    plt.show()
                    plt.clf()
        return Z
        
    def end(self):
        pass


       
    