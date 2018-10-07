from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal as mvn
import random
import copy
import cProfile
import re
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import os; 
from math import sqrt
import signal
import sys
import cProfile
sys.path.append('../src/'); 
from gaussianMixtures import Gaussian
from gaussianMixtures import GM
import matplotlib.animation as animation
from numpy import arange
import time
import matplotlib.image as mgimg
from softmaxModels import Softmax

'''
****************************************************
File: D10MulitRobotModel.py
Written By: Luke Burks
October 2018

Container Class for problem specific models
Model: 5 2D Robots, each trying to reach a specific 
location
The "Multi-Robot 2D Hallway Problem"

Bounded 0,10 for each dimension

****************************************************
'''


class ModelSpec:

	def __init__(self):
		self.fileNamePrefix = 'D10MultiRobot'; 
		self.STM = None
		self.acts = 26; 
		self.obs = 84; 


	def buildTransition(self):
		self.bounds = []; 
		for i in range(0,10):
			self.bounds.append([0,10]); 

		#Need a separate 10x10 matrix for each action...

		delta = 1;
		self.delA = []; 
		self.discount = 0.95; 