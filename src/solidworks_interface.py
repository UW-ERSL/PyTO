
import os
import win32com.client #pip install pypiwin32
from stl_reader import STLGeom
import numpy as np

class SolidWorksInterface:
    def __init__(self):
        try:
            self.sw = win32com.client.Dispatch("SldWorks.Application")  
            self.part = self.sw.ActiveDoc
        except Exception as e:
            print(f"Error: {str(e)}")

    def getEdgeCount(self):
        try:
            bodies = self.part.GetBodies2(0,True)
            edgeCount = bodies[0].GetEdgeCount()
            return edgeCount
        except Exception as e:
            print(f"Error: {str(e)}")
            return 0
    
    def getFaceCount(self):
        try:
            bodies = self.part.GetBodies2(0,True)
            faceCount = bodies[0].GetFaceCount()
            return faceCount
        except Exception as e:
            print(f"Error: {str(e)}")
            return 0
        
    def getAllDimensions(self):
        try:
            model = self.sw.ActiveDoc
            swConfig = model.ConfigurationManager.ActiveConfiguration
            parameterCount = swConfig.GetParameterCount
            print(f"Number of parameters: {parameterCount}")
            dimensions = []
            names = []
            values = []
            swConfig.GetParameters(parameterCount,names, values)
            dimensions = list(zip(names, values))
            print(dimensions)
            input("Press Enter to continue...")
            return dimensions
        except Exception as e:
            print(f"Error: {str(e)}")
            input("Press Enter to continue...")
            return []

    def getDimension(self,dimensionName):
        try:
            model = self.sw.ActiveDoc
            myDimension = model.Parameter(dimensionName)
            
            return myDimension.SystemValue
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
        
    def setDimension(self,dimensionName, value):
        try:
            model = self.sw.ActiveDoc
            myDimension = model.Parameter(dimensionName)
            myDimension.SystemValue = value
            model.ForceRebuild3(True) 
            dimCheck = self.getDimension(dimensionName)  # get dimension
            if not np.isclose(dimCheck, value, atol=1e-6):
                #print(f"Dimension {dimensionName} not set correctly. Expected {value}, got {dimCheck}.")
                return False
            print(f"Dimension {dimensionName} set to {value}.")
            if not self.isBodyValid():
                #print("Body is invalid after dimension change.")
                return False
            return True
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
        
    def setDimensions(self,dimensionNames, values):
        try:
            model = self.sw.ActiveDoc
            for dimensionName, value in zip(dimensionNames, values):
                myDimension = model.Parameter(dimensionName)
                myDimension.SystemValue = value
            model.ForceRebuild3(True) 
            for dimensionName, value in zip(dimensionNames, values): # check if dimensions are set correctly
                dimCheck = self.getDimension(dimensionName)  # get dimension
                if not np.isclose(dimCheck, value, atol=1e-6):
                    print(f"Dimension {dimensionName} not set correctly. Expected {value}, got {dimCheck}.")
                    return False
                print(f"Dimension {dimensionName} set to {value}.")
            if not self.isBodyValid():
                #print("Body is invalid after dimension change.")
                return False
            return True
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
        
    def getMassProperties(self):
        try:
            model = self.sw.ActiveDoc
            body = model.GetBodies2(0, True)
            
            massProp = body[0].GetMassProperties(0)
            cg = [x for x in massProp[0:3]]
            volume = massProp[3]
            area = massProp[4]
            inertia = [x for x in massProp[6:12]]
            return [ area, volume, cg, inertia]
        
        except Exception as e:  
            print(f"Error: {str(e)}")
            return False  
        
    def isBodyValid(self):
        try:
            model = self.sw.ActiveDoc
            body = model.GetBodies2(0, True)
            swFaultEnt = body[0].Check3
            count = swFaultEnt.Count
            if count == 0:
                return True
            else:
                return False
        except Exception as e:  
            print(f"Error: {str(e)}")
            return False
     
    def getSTL(self):
        try:
            self.doc = self.sw.ActiveDoc
            if self.doc is None:
                print("No document is open in SolidWorks")
                return False
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.stl_file = os.path.join(script_dir, './temp.STL')
            self.doc.SaveAs3(self.stl_file, 0, 2)
            self.stlGeom = STLGeom(self.stl_file)
            os.remove(self.stl_file)
            return self.stlGeom
        except Exception as e:
            print(f"Error: {str(e)}")
            return False

    def solveFEAStudy(self):
        try:
            print("Does not work")
            return
            cw = self.sw.GetAddInObject("SldWorks.Simulation")
            sm = cw.COSMOSWORKS.ActiveDoc.StudyManager
            s = sm.GetStudy(0)
            returnCode = s.RunAnalysis
            print(f"Return Code: {returnCode}")
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
        
if __name__ == "__main__":
    input("Is SolidWorks open with a threeHolesPart?\nPress Enter to continue...")
    sw = SolidWorksInterface()
   

    dimName = "d@SideHolesSketch"
    dim0 = sw.getDimension(dimName)  # get dimension
    [area,vol,cg,inertia] = sw.getMassProperties()
    print(f"\nProperties Before:")
    print(f"Dim    : {dim0:.3e}")
    print(f"Area    : {area:.3e}")
    print(f"Volume  : {vol:.3e}")
    print(f"CG      : [{cg[0]:.3e}, {cg[1]:.3e}, {cg[2]:.3e}]")
    print(f"Inertia : [{', '.join([f'{x:.3e}' for x in inertia])}]")
    print(f"Edge Count: {sw.getEdgeCount()}")
    print(f"Face Count: {sw.getFaceCount()}")
    
    if not sw.setDimension(dimName,1.1*dim0):
        print(f"Failed to set dimension. Resetting to original value.")
        sw.setDimension(dimName,dim0)

    dim = sw.getDimension(dimName)  # get dimension
  
    [area,vol,cg,inertia] = sw.getMassProperties()
    print(f"\nProperties After:")
    print(f"Dim    : {dim:.3e}")
    print(f"Area    : {area:.3e}")
    print(f"Volume  : {vol:.3e}")
    print(f"CG      : [{cg[0]:.3e}, {cg[1]:.3e}, {cg[2]:.3e}]")
    print(f"Inertia : [{', '.join([f'{x:.3e}' for x in inertia])}]")
    print(f"Edge Count: {sw.getEdgeCount()}")
    print(f"Face Count: {sw.getFaceCount()}")
    input("Press Enter to continue...")
    sw.setDimension(dimName,dim0)  # reset
 