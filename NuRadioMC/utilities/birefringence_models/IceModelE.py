import numpy as np
import csv
from scipy import constants
from scipy import optimize
from scipy import interpolate
#import scipy.optimize
import matplotlib.pyplot as plt
import time
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioMC.utilities import medium
from NuRadioMC.utilities.medium import southpole_2015
from NuRadioReco.utilities import units
import logging
from sklearn.feature_selection.tests.test_from_model import test_allow_nan_tag_comes_from_estimator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('raytracing')


model = 'E'


c = constants.c
e_p = 3.157                 #given in Jordan
e_d = 0.034                 #given in Jordan

m = southpole_2015()  
comp = m.get_index_of_refraction(np.array([0, 0, -2500]))



def get_index_of_refraction(z):
    

    
    n1 = m.get_index_of_refraction(z) + f1_rec(-z[2]) - comp
    n2 = m.get_index_of_refraction(z) + f2_rec(-z[2]) - comp
    n3 = m.get_index_of_refraction(z) + f3_rec(-z[2]) - comp 
            
    return n1, n2, n3


def get_index_of_refraction_all(z):

    """
    Returns the three indices of refraction for every meter down to the specified depth.
    
    Parameters
    ---------
    z:    numpy.array (int as entries), position of the interaction (only the z-coordinate matters)
    
    n:    numpy.array, every entry is a list with the three indices of refrection at the specific depth
      
    """
            
    n = []
    znew = np.arange(z[2], 0, 1)
    
    #print(znew)
    
    pos = np.zeros((len(znew), 3))       
    pos[:,2] = znew
    
    #print(pos)
                         
    for i in znew:

        n.append(get_index_of_refraction(pos[i]))
    
    return(np.array(n), znew)



depthL = []
e1 = []
e2 = []
e3 = []

with open("eps.csv", 'r') as inputfile:
    for i in range(84):
        line = inputfile.readline()
        if i > 1:
            line_s = line.split(',')
            depthL.append(float(line_s[0]))
            e3.append(float(line_s[3]))
            e2.append(float(line_s[4]))
            e1.append(float(line_s[5])) 


dep = np.array(depthL) 
e1 = np.array(e1)
e2 = np.array(e2)
e3 = np.array(e3)

n1 = np.sqrt(e_p + e_d * e1)
n2 = np.sqrt(e_p + e_d * e2)
n3 = np.sqrt(e_p + e_d * e3)
depth = dep

n1 = np.array(n1)
n2_av = np.array(n2)
n3_av = np.array(n3)

print(depth)

n2 = (n2_av+n3_av)/2
n3 = (n2_av+n3_av)/2
#n2, n3 = n2 + n3 /2

long = 1000

filler_n1 = np.full((1,long), np.mean(n1[-10:]))
filler_n2 = np.full((1,long), np.mean(n2[-10:]))
filler_n3 = np.full((1,long), np.mean(n3[-10:]))
filler_d = np.linspace(1800, 2500,long)

n1 = np.concatenate((n1, filler_n1), axis=None)
n2 = np.concatenate((n2, filler_n2), axis=None)
n3 = np.concatenate((n3, filler_n3), axis=None)

depth = np.concatenate((depth, filler_d), axis=None)

filler_n1 = np.full((1,long), np.mean(n1[:10]))
filler_n2 = np.full((1,long), np.mean(n2[:10]))
filler_n3 = np.full((1,long), np.mean(n3[:10]))
filler_d = np.linspace(0, 140,long)

n1 = np.concatenate((filler_n1, n1), axis=None)
n2 = np.concatenate((filler_n2, n2), axis=None)
n3 = np.concatenate((filler_n3, n3), axis=None)

depth = np.concatenate((filler_d, depth), axis=None)


#n2 = n2 + n3 / 2


#print(n3)

#xnew = np.arange(depth[0], depth[-1], 20)
xnew = np.arange(0, 2500, 1)

#node_cond = 0.0000015           # s < len(n) * np.var(n)

#wrinkly interpolation
node_cond1 = 0.0000007
node_cond2 = 0.0000015
node_cond3 = 0.000001


#smooth interpolation
node_cond1 = 0.000001
node_cond2 = 0.0000003
node_cond3 = 0.0000003

f1 = interpolate.UnivariateSpline(depth, n1, s = node_cond1)     
f2 = interpolate.UnivariateSpline(depth, n2, s = node_cond2)         
f3 = interpolate.UnivariateSpline(depth, n3, s = node_cond3)

tck1 = f1._eval_args
f1_rec = interpolate.UnivariateSpline._from_tck(tck1)

tck2 = f2._eval_args
f2_rec = interpolate.UnivariateSpline._from_tck(tck2)

tck3 = f3._eval_args
f3_rec = interpolate.UnivariateSpline._from_tck(tck3)



interpolation = np.array([tck1, tck2, tck3])

if 0:
    np.save('index_model' + model + '.npy', interpolation)


#--------------Jordan interpolation

if 1:
    
    n1 = n1[(depth<=1739)&(depth>=140)]
    n2 = n2[(depth<=1739)&(depth>=140)]
    n3 = n3[(depth<=1739)&(depth>=140)]
    depth = depth[(depth<=1739)&(depth>=140)]
    
    
    plt.plot(-depth, n1, 'b.', label = 'nx - data')
    plt.plot(-xnew, f1(xnew), label = 'nx - interpolation')
    
    plt.plot(-depth, n2, 'c.', label = '(ny + nz)/2 - data')
    plt.plot(-xnew, f2(xnew), label = '(ny + nz)/2 - interpolation')

    
    plt.plot(-dep, n2_av, 'r.', label = 'ny - data')
    #plt.plot(-xnew, f2(xnew), label = 'ny - interpolation')
    
    plt.plot(-dep, n3_av, 'g.', label = 'nz - data')
    #plt.plot(-xnew, f3(xnew), label = 'nz - interpolation')    

    
    plt.title('Principle refractive index at SPICE')
    plt.xlabel('depth [m]')
    plt.ylabel('refractive index')
    plt.legend()
    #plt.xlim(-1750, -140)
    #plt.ylim(1.778, 1.782)
    plt.grid()
    plt.show()



#-------------Jordan + Southpole
if 0:
    p = [0, 0, -2500]
    
    i_all = get_index_of_refraction_all(p)
    
    
    
    plt.plot(i_all[1], i_all[0][:,0], label = 'nx - model')
    plt.plot(i_all[1], i_all[0][:,1], label = 'ny - model')
    plt.plot(i_all[1], i_all[0][:,2], label = 'nz - model')
    
    plt.title('refractive index for birefringence')
    plt.xlabel('depth [m]')
    plt.ylabel('refractive index')
    plt.legend()
    #plt.xlim(-1750, -140)
    #plt.ylim(1.778, 1.782)
    plt.grid()
    plt.show()
    
    







































