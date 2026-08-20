from solidworks_interface import SolidWorksInterface 
from stl_reader import STLGeom
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sw = SolidWorksInterface()
fileName = "temp.stl"
sw.saveSTL(fileName)
stl_file = os.path.join(script_dir, fileName)
stl = STLGeom(stl_file)
stl.plotGeometry()