# -*- coding: utf-8 -*-
# QuickFF is a code to quickly derive accurate force fields from ab initio input.
# Copyright (C) 2012 - 2016 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
# Steven Vandenbrande <Steven.Vandenbrande@UGent.be>,
# Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
#
# This file is part of QuickFF.
#
# QuickFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# QuickFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--
from molmod.units import *
from quickff.tools import boxqp
import numpy as np

__all__ = ['HessianFCCost']

class HessianFCCost(object):
    def __init__(self, system, ai, valence, fit_indices, ffrefs=[]):
        '''
            **Arguments**
            
            system
                a Yaff system object
            
            ai
                an instance of the Reference representing the ab initio input
            
            valence
                A ValenceFF object containing all valence terms.
            
            fit_indices
                a list of indices indicating the terms for which the force
                constants should be determined.
            
            **Optional Arguments**
                        
            ffrefs
                a list of Reference instances representing possible a priori
                determined contributions to the force field (such as eg. 
                electrostatics and van der Waals)
        '''
        #initialization
        self.init = np.zeros(len(fit_indices), float)
        self.upper = np.zeros(len(fit_indices), float)+np.inf
        self.lower = np.zeros(len(fit_indices), float)
        self.A = np.zeros([len(fit_indices), len(fit_indices)], float)
        self.B = np.zeros([len(fit_indices)], float)
        ndofs = 3*system.natom
        #compute the reference hessian
        href = ai.phess0.reshape([ndofs, ndofs]).copy()
        for ffref in ffrefs:
            href -= ffref.hessian(system.pos).reshape([ndofs, ndofs])
        #loop over valence terms and add to reference (if not in fit_indices or
        #its slaves) or add to covalent hessians hcovs (if in fit_indices)
        hcovs = [None,]*len(fit_indices)
        for master in valence.iter_masters():
            if master.index in fit_indices:
                i = fit_indices.index(master.index)
                #add to covalent hessians (includes slaves as well)
                hcov = valence.get_hessian_contrib(master.index, fc=1.0)
                hcovs[i] = hcov
                #set upper and lower
                if master.kind==4:
                    self.upper[i] = 200*kjmol
                if master.kind==3:
                    self.lower[i] = -np.inf
            else:
                hcov = valence.get_hessian_contrib(master.index)
                href -= hcov
        #construct the cost matrices A and B
        for index1, hcov1 in enumerate(hcovs):
            self.B[index1] = np.sum(href*hcov1)
            self.A[index1,index1] = np.sum(hcov1*hcov1)
            for index2, hcov2 in enumerate(hcovs[:index1]):
                tmp = np.sum(hcov1*hcov2)
                self.A[index1,index2] = tmp
                self.A[index2,index1] = tmp
    
    def estimate(self, init=None, lower=None, upper=None):
        '''
            Estimate the force constants by minimizing the cost function

        '''
        if init is None:
            assert self.init is not None, 'No initial fcs defined'
            init = self.init.copy()
        if lower is None:
            assert self.lower is not None, 'No lower limit fcs defined'
            lower = self.lower.copy()
        if upper is None:
            assert self.upper is not None, 'No upper limit fcs defined'
            upper = self.upper.copy()
        U, S, Vt = np.linalg.svd(self.A)
        #print '  Singular Values of cost.A: ', S
        #print ' Vt=', Vt
        x = boxqp(self.A, self.B, lower, upper, init)
        return x
