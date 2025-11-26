import numpy as np
import mat_lib
import bound_cond
import hex_mesher
import os
import enum
import scipy.sparse as spy_sprs
from stl_reader import STLGeom
from scipy.sparse import lil_matrix
script_dir = os.path.dirname(os.path.abspath(__file__))


class ThermoStructuralExamples(enum.Enum):
	TensileBar = enum.auto()
	
def getThermoStructuralProblem(problem: ThermoStructuralExamples, **kwargs):
	if problem == ThermoStructuralExamples.TensileBar:
		return tensileBarExample(**kwargs)
	else:
		raise ValueError("Unknown problem type.")
	
def tensileBarExample():