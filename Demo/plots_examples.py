
import enum
import pyvista as pv

class ExamplesCAD(enum.Enum):
	EdgeCantileverDemo = enum.auto()
	BliskSectionWithBlade = enum.auto()
	KnuckleAssembly = enum.auto()
	Mitchell_1 = enum.auto()
	ShortCantileverMidLoad = enum.auto()
	CantileverMidLoad = enum.auto()
	TwoBar = enum.auto()
	MBBB = enum.auto()
	DistributedLoad = enum.auto()
	LBracketMidLoad = enum.auto()
	VerticalBar = enum.auto()
	FilletedBeam = enum.auto()
	ThreeHoleBracket = enum.auto()
	CircularPlateHole = enum.auto()
	TorquePlate = enum.auto()
	CentrifugalPlate = enum.auto()
	BasePlateOptimizableVol = enum.auto()
  
  

   
  
def get_example_cad(example: ExamplesCAD, plot: bool = False):
  fp_stl_folder = "../Models/"
  fp_vtu_mesh_folder = "../CadRecoveryResults/"
  fp_output_folder = "../CadRecoveryResults/results/"
  if example == ExamplesCAD.EdgeCantileverDemo:
    str_output_name = "EdgeCantileverDemo"
    fp_original_stl = fp_stl_folder + "EdgeCantilever/EdgeCantilever.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.BliskSectionWithBlade:
    fp_original_stl = fp_stl_folder + "Saketh/BliskSectionWithBlade2test.STL"
    fp_vtu_mesh = "../Models/Saketh/test.vtu"
    fp_outputstlpath = "../Models/Saketh/BliskSectionWithBlade2Recovered.stl"
    fp_outputfixedstlpath = "../Models/Saketh/BliskSectionWithBlade2RecoveredFixed.stl"
  elif example == ExamplesCAD.KnuckleAssembly:
    fp_original_stl = fp_stl_folder + "KnuckleAssembly/KnuckleAssembly.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + "KnuckleAssembly (2).vtu" 
    fp_outputstlpath = fp_output_folder + "KnuckleAssemblyRecovered.stl"
    fp_outputfixedstlpath = fp_output_folder + "KnuckleAssemblyRecoveredFixed.stl"
  elif example == ExamplesCAD.Mitchell_1:
    str_output_name = "Mitchell_1"
    fp_original_stl = fp_stl_folder + "Mitchell/Mitchell.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.ShortCantileverMidLoad:
    str_output_name = "ShortCantileverMidLoad"
    fp_original_stl = fp_stl_folder + "ShortCantilever/ShortCantilever.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.CantileverMidLoad:
    str_output_name = "CantileverMidLoad"
    fp_original_stl = fp_stl_folder + "Cantilever/Cantilever.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.TwoBar:
    str_output_name = "TwoBar"
    fp_original_stl = fp_stl_folder + "TwoBar/TwoBar.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.MBBB:
    str_output_name = "MBBB"
    fp_original_stl = fp_stl_folder + "MBBB/MBBB.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.DistributedLoad:
    str_output_name = "DistributedLoad"
    fp_original_stl = fp_stl_folder + "DistributedLoad/DistributedLoad.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.LBracketMidLoad:
    str_output_name = "LBracketMidLoad"
    fp_original_stl = fp_stl_folder + "LBracket/LBracket.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.VerticalBar:
    str_output_name = "VerticalBar"
    fp_original_stl = fp_stl_folder + "VerticalBar/VerticalBar.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.FilletedBeam:
    str_output_name = "FilletedBeam"
    fp_original_stl = fp_stl_folder + "FilletedBeam/FilletedBeam.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.ThreeHoleBracket:
    str_output_name = "ThreeHoleBracket"
    fp_original_stl = fp_stl_folder + "ThreeHoleBracket/ThreeHoleBracket.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'  
  elif example == ExamplesCAD.CircularPlateHole:
    str_output_name = "CircularPlateHole"
    fp_original_stl = fp_stl_folder + "CircularPlateHole/CircularPlateHole.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl' 
  elif example == ExamplesCAD.TorquePlate:
    str_output_name = "TorquePlate"
    fp_original_stl = fp_stl_folder + "CircularPlateHole/CircularPlateHole.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.CentrifugalPlate:
    str_output_name = "CentrifugalPlate"
    fp_original_stl = fp_stl_folder + "CircularPlateHole/CircularPlateHole.STL"
    fp_vtu_mesh = fp_vtu_mesh_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  elif example == ExamplesCAD.BasePlateOptimizableVol:
    str_output_name = "BasePlateOptimizableVol"
    fp_stl_folder = fp_stl_folder + "Rocket/BasePlateOptimizableVol/"
    fp_original_stl = fp_stl_folder + "BasePlateOptimizableVol.STL"
    fp_vtu_mesh = fp_stl_folder + f'{str_output_name}.vtu'
    fp_outputstlpath = fp_output_folder + f'{str_output_name}Recovered.stl'
    fp_outputfixedstlpath = fp_output_folder + f'{str_output_name}RecoveredFixed.stl'
  else:
    raise ValueError(f"Unknown example: {example}")
  
  if plot:
    meshA = pv.read(fp_original_stl)
    meshB = pv.read(fp_vtu_mesh)
    # --- 4) Visualize everything together in one Plotter ---
    pl = pv.Plotter()
    pl.add_text("Marching Cubes Reconstruction", font_size=14)
    pl.add_mesh(meshA, color="lightblue")
    #pl.add_mesh(meshB, color="red")
    pl.show()

  return fp_original_stl, fp_vtu_mesh, fp_outputstlpath, fp_outputfixedstlpath


if __name__ == "__main__":    
   
  example = ExamplesCAD.BliskSectionWithBlade 
  input_stl, input_vtu, output_stl, out_stl_fixed = get_example_cad(example, True)
  pv.read(output_stl).plot()