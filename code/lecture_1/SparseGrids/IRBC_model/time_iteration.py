"""
This file contains routines that are used within the time iteration procedure.
In particular, it consists of:

	- the time iteration step at the beginning of each iteration
	- the grid refinement process
	- the construction of a new grid
	- the calculation of the convergence criterion
	- the storage of the grid structure

"""



################################################################################
#                        Load necessary libraries                              #
################################################################################

import numpy as np
import Tasmanian

from parameters import *
from scipy import optimize
from Expect_FOC import *
from SOE import *
from setup_asg import *


################################################################################
#                           Time iteration step                                #
################################################################################

def ti_step(grid,pol_guess,gridZero):

    # Get the points that require function values
    aPoints1 = grid.getNeededPoints()
    # Get the number of points that require function values
    aNumAdd = grid.getNumNeeded()

    # Array for intermediate update step
    polInt = np.zeros((aNumAdd,nPols))

    # Time Iteration step
    for ii1 in range(aNumAdd):

        state = aPoints1[ii1]
        pol = pol_guess[ii1,:]
        root = optimize.root(sysOfEqs, pol, args=(state,gridZero), method='hybr')
        polInt[ii1,:] = root.x

    # Add the new function values to grid1
    grid.loadNeededPoints(polInt)


    return grid



################################################################################
#                             Grid refinement                                  #
################################################################################

def refine(grid):

    # Get the points that require function values
    aNumLoad = grid.getNumLoaded()
    # Scaling to only allow for those policies that are supposed to
    # determine the refinement process (in this case only the capital policies)
    scaleCorrMat = np.zeros((aNumLoad,nPols))
    scaleCorrMat[:,0:nPols+1] = scaleCorr

    # Refine the grid based on the surplus coefficients
    grid.setSurplusRefinement(surplThreshold, dimRef, typeRefinement, [], scaleCorrMat)

    if (grid.getNumNeeded()>0):

	    # Get the new points and the number of points
	    nwpts = grid.getNeededPoints()
	    aNumNew = grid.getNumNeeded()

	    # We assign (for now) function values through interpolation#
	    pol_guess = np.zeros((aNumNew,nPols))
	    pol_guess = grid.evaluateBatch(nwpts)

    else:

	    pol_guess = []


    return grid, pol_guess



################################################################################
#                        New grid construction                                 #
################################################################################

def fresh_grid():

    # Generate the grid structure
    grid = Tasmanian.makeLocalPolynomialGrid(gridDim,gridOut,gridDepth,gridOrder,gridRule)
    # Transform the domain
    grid.setDomainTransform(gridDomain)

    return grid



################################################################################
#                Checking convergence and updating policies                    #
################################################################################

def policy_update(gridOld,gridNew):

    # Get the points and the number of points from grid1
    aPoints2 = gridNew.getPoints()
    aNumTot = gridNew.getNumPoints()

    # Evaluate the grid points on both grid structures
    polGuessTr1 = gridNew.evaluateBatch(aPoints2)
    polGuessTr0 = gridOld.evaluateBatch(aPoints2)

    # 1) Compute the Sup-Norm

    metricAux = np.zeros(nPols)

    for imet in range(nPols):
        metricAux[imet] = np.amax(np.abs(polGuessTr0[:,imet]-polGuessTr1[:,imet]))

    metricSup = np.amax(metricAux)

    # 2) Compute the L2-Norm

    metricL2 = 0.0

    for imetL2 in range(nPols):
        metricL2 += np.sum((np.abs(polGuessTr0[:,imetL2]-polGuessTr1[:,imetL2]))**2)

    metricL2 = (metricL2/(aNumTot*nPols))**0.5

    metric = np.minimum(metricL2,metricSup)

    # Now update pol_guess and grid

    polGuess = np.zeros((aNumTot,nPols))

    for iupd in range(nPols):
        polGuess[:,iupd] = 0.5*polGuessTr0[:,iupd] + 0.5*polGuessTr1[:,iupd]

    gridOld = Tasmanian.copyGrid(gridNew)


    return metric, polGuess, gridOld


################################################################################
#                               Grid storage                                   #
################################################################################

def save_grid(grid,iter):

    if typeIRBC=='non-smooth':
        grid.write(data_location_nonsmooth + "grid_iter_" + str(iter+1) + ".txt")
    else:
        grid.write(data_location_smooth + "grid_iter_" + str(iter+1) + ".txt")


    return
