
import os
import win32com.client #pip install pypiwin32
from STLGeom import STLGeom

class SolidWorksInterface:
    def __init__(self):
        try:
            self.sw = win32com.client.Dispatch("SldWorks.Application")  
        except Exception as e:
            print(f"Error: {str(e)}")

    def getSTLofActiveSolidWorksModel(self):
        try:
            self.doc = self.sw.ActiveDoc
            if self.doc is None:
                print("No document is open in SolidWorks")
                return False
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.stl_file = os.path.join(script_dir, './temp.STL')
            self.doc.SaveAs3(self.stl_file, 0, 2)
            self.stlGeom = STLGeom(self.stl_file)
            return self.stlGeom
        except Exception as e:
            print(f"Error: {str(e)}")
            return False

    def __del__(self):
        os.remove(self.stl_file)


if __name__ == "__main__":
    sw = SolidWorksInterface()
    stlGeom = sw.getSTLofActiveSolidWorksModel()
    if stlGeom:
        stlGeom.plotGeometry()
    del sw