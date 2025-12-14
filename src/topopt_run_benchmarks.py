from topopt_thermostructural_benchmarks import ThermoStructuralTOExamples
from topopt_common import *
from topopt_mma import topopt_mma
from topopt_ocm import topopt_optimality_criteria	
from topopt_pareto import topopt_pareto

from topopt_levelset import topopt_levelset	
from topopt_structural_benchmarks import *
from topopt_thermal_benchmarks import *
from topopt_thermostructural_benchmarks import *
import time
import glob
import pandas as pd

def runTOMethodOnBenchmarks(optimizationMethod):
	# Create a list to store results

	saveVTU = False  # Set to True if you want to save the VTU files for MMA method
	binarize_topology = True  # Set to True if you want to binarize the topology for MMA/OCM method
	results_list = []
	dsolver = deflation.DeflationSolver()
	feaMode = FEA_MODE.STRUCTURAL
	benchmarks_structural_2_5D_problems_1 = [StructuralTOExamples.Mitchell_1, 
							StructuralTOExamples.CantileverTipLoad, 
							StructuralTOExamples.CantileverMidLoad,
							StructuralTOExamples.MBBBeam,
							StructuralTOExamples.Bridge,
							StructuralTOExamples.TwoBar,]
	
	benchmarks_structural_2_5D_problems_2 = [StructuralTOExamples.LBracketTopLoad, 
							StructuralTOExamples.LBracketMidLoad,
							StructuralTOExamples.TorquePlate,
							StructuralTOExamples.DistributedLoad,
							StructuralTOExamples.ThreeHoleBracket,]

	benchmarks_structural_3D_problems = [StructuralTOExamples.EdgeCantilever, 
							StructuralTOExamples.ThreeHoleBracketThick, 
							StructuralTOExamples.Multiload,
							StructuralTOExamples.LBracketThickTopLoad,
							StructuralTOExamples.LBracketThickMidLoad,
							StructuralTOExamples.Table,
							StructuralTOExamples.GEGrabCAD,]
	
	benchmarks_noncompliance_problems = [StructuralTOExamples.CantileverMidLoadVolumeCompliance,
						StructuralTOExamples.LBracketTopLoad_Stress_Vol, 
						StructuralTOExamples.LBracketTopLoad_Vol_Stress, 
						StructuralTOExamples.LBracketThickTopLoad_Vol_Stress,
						StructuralTOExamples.LBracketMidLoad_Vol_Stress,
						StructuralTOExamples.LBracketTopLoad_Mass_StressFF,
						StructuralTOExamples.Inverter]
		
	benchmarks_bodyforce_problems = [StructuralTOExamples.GravityPlate,
						StructuralTOExamples.CentrifugalPlate]

	benchmarks_thermostructural_problems = [ThermoStructuralTOExamples.BiClamp,
						ThermoStructuralTOExamples.MBBBeam]
	
	benchmarks_thermal_2_5_problems = [ThermalTOExamples.HeatPlate, ThermalTOExamples.FourCornersThermal,
							 ThermalTOExamples.BridgeThermal]	
	
	sampleBenchmarks = [StructuralTOExamples.Mitchell_1, StructuralTOExamples.LBracketTopLoad_Stress_Vol,
						StructuralTOExamples.EdgeCantilever,StructuralTOExamples.GravityPlate,ThermoStructuralTOExamples.MBBBeam,
						ThermalTOExamples.FourCornersThermal]
	
	allBenchmarks = benchmarks_structural_2_5D_problems_1 + \
					benchmarks_structural_2_5D_problems_2 + \
					benchmarks_thermal_2_5_problems + \
					benchmarks_structural_3D_problems  + \
					benchmarks_noncompliance_problems + \
					benchmarks_bodyforce_problems + \
					benchmarks_thermostructural_problems
	
	for to_problem in  benchmarks_structural_2_5D_problems_2:  
		if to_problem in benchmarks_structural_2_5D_problems_1:
			subFolder = "Structural-Compliance2.5D_1"
		elif to_problem in benchmarks_structural_2_5D_problems_2:
			subFolder = "Structural-Compliance2.5D_2"
		elif to_problem in benchmarks_thermal_2_5_problems:
			subFolder = "Thermal-Compliance2.5D"
		elif to_problem in benchmarks_structural_3D_problems:
			subFolder = "Structural-Compliance3D"
		elif to_problem in benchmarks_noncompliance_problems:
			subFolder = "Structural-NonCompliance"
		elif to_problem in benchmarks_bodyforce_problems:
			subFolder = "Structural-BodyForce"
		elif to_problem in benchmarks_thermostructural_problems:
			subFolder = "ThermoStructural"
		else:
			subFolder = "Other"

		if (to_problem in StructuralTOExamples):
			mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
			feaMode = FEA_MODE.STRUCTURAL
		elif (to_problem in ThermalTOExamples):
			mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)
			feaMode = FEA_MODE.THERMAL
		elif (to_problem in ThermoStructuralTOExamples):
			mesh, mat_prop, structural_bc,thermal_bc, elem_body_force, to_params  = getThermoStructuralTOProblem(to_problem)
			feaMode = FEA_MODE.THERMO_STRUCTURAL # or FEA_MODE.STRUCTURAL depending on the problem setup

		
		print_progress = False
		fe_structural_solver = None
		fe_thermal_solver = None

		dsolver = deflation.DeflationSolver()
		if (to_params.nDOFDesired <= DIRECT_SOLVER_DOF_CUTOFF):#  # Choose solver. Typically PARDISO, but DPCG for large DOF problems
			solver = lin_solv.Solvers.PARDISO
		else:
			solver = lin_solv.Solvers.DPCG
			nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
			dsolver.create_deflation_groups(mesh, nGroups)
			dsolver.create_deflation_matrix(mesh)
			dsolver.W = dsolver.W[bc.free_dofs, :]

		if (feaMode == FEA_MODE.STRUCTURAL):
			fe_structural_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
						mat_prop = mat_prop,
						bc = bc,
						solver = solver,
						dsolver = dsolver,
						rtol = 1e-8,
						elem_body_force = elem_body_force)
			fe_solver = fe_structural_solver

		#fe_structural_solver.plot_mesh(title = "Structural Load", plot_bc = True, save_path = None)
		elif (feaMode == FEA_MODE.THERMAL):
			fe_thermal_solver = hex_thermal_fea.HexThermalFEA(mesh = mesh,
						mat_prop = mat_prop,
						bc = bc,
						solver = solver,
						dsolver = dsolver,
						rtol = 1e-8,
						elem_body_force = elem_body_force)
			fe_solver = fe_thermal_solver

		#fe_thermal_solver.plot_mesh(title = "Thermal Load", plot_bc = True, save_path = None)
		elif (feaMode == FEA_MODE.THERMO_STRUCTURAL):
			fe_structural_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
						mat_prop = mat_prop,
						bc = structural_bc,
						solver = solver,
						dsolver = dsolver,
						rtol = 1e-8,
						elem_body_force = elem_body_force)
			fe_thermal_solver = hex_thermal_fea.HexThermalFEA(mesh = mesh,
						mat_prop = mat_prop,
						bc = thermal_bc,
						solver = solver,
						dsolver = dsolver,
						rtol = 1e-8)
			fe_solver = fe_structural_solver  # primary solver is structural
			
		# Ensure the output directory exists
		output_base = f"./Results/Results_{time.strftime('%Y-%m-%d')}/{subFolder}/Problems/"
		os.makedirs(output_base, exist_ok=True)

		if to_problem in benchmarks_structural_2_5D_problems_1 or to_problem in benchmarks_structural_2_5D_problems_2 or to_problem in benchmarks_thermal_2_5_problems:
			fe_solver.plot_mesh(
				title="Structural Load",
				plot_bc=True,
				camera_position='xy',
				save_path=f"{output_base}/{to_problem.name}_problem.png"
			)
		else:
			fe_solver.plot_mesh(
				title="Structural Load",
				plot_bc=True,
				save_path=f"{output_base}/{to_problem.name}_problem.png"
			)
		startTime = time.time()
		print("-" * 50)
		print(f"Running {to_problem.name} problem using {optimizationMethod.name} method and {solver.name} solver")
		print("-" * 50)
		# Create the directory if it does not exist
		output_dir = f"./Results/Results_{time.strftime('%Y-%m-%d')}/{subFolder}/{optimizationMethod.name}"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		
		if optimizationMethod == TO_METHODS.DENSITYMMA:
			u, history,success,errorMsg,nFEAs = topopt_mma(feaMode, fe_structural_solver,fe_thermal_solver,binarize_topology = binarize_topology,
									to_params = to_params,print_progress = print_progress)
		elif optimizationMethod == TO_METHODS.DENSITYOCM:
			if to_problem in benchmarks_noncompliance_problems or \
					to_problem in benchmarks_bodyforce_problems or \
					to_problem in benchmarks_thermostructural_problems:
				continue
			u, history, success,errorMsg,nFEAs = topopt_optimality_criteria(feaMode, fe_solver,binarize_topology = binarize_topology,
											to_params = to_params,print_progress = print_progress)
		elif optimizationMethod == TO_METHODS.PARETO:
			if to_problem in benchmarks_noncompliance_problems or \
					to_problem in benchmarks_bodyforce_problems or \
					to_problem in benchmarks_thermostructural_problems:
				continue
			u, history, success,errorMsg,nFEAs = topopt_pareto(feaMode, fe_solver,
													to_params = to_params,print_progress = print_progress)
		elif optimizationMethod == TO_METHODS.LEVELSET:
			if to_problem in benchmarks_noncompliance_problems or \
					to_problem in benchmarks_bodyforce_problems or \
					to_problem in benchmarks_thermostructural_problems:
				continue
			u, history, success,errorMsg,nFEAs = topopt_levelset(feaMode,  fe_solver,
													to_params = to_params)
		timeTaken = time.time() - startTime

		image_path = f"{output_dir}/{to_problem.name}.png"
		title = f"{optimizationMethod.name}: vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"
	
		if to_problem in benchmarks_structural_2_5D_problems_1 or to_problem in benchmarks_structural_2_5D_problems_2 or to_problem in benchmarks_thermal_2_5_problems:
			fe_solver.plot_mesh(save_path=image_path, plot_bc = None, title=title, camera_position='xy')
		else:
			fe_solver.plot_mesh(save_path=image_path, plot_bc = None, title=title)
	
		results_list.append({
			'name': to_problem.name,
			'comment': to_params.Comment,  
			'ndof': 3*fe_solver.mesh.num_nodes,
			'volfrac': history['volfrac'][-1],
			'objective': history['objective'][-1],
			'#FEAs': nFEAs,
			'time (s)': timeTaken,
			'success': success,
			'errorMsg': errorMsg
		})
		# Check if a previous CSV result exists for this method and problem
		result_csv_file = f"{output_dir}/{optimizationMethod.name}_summary.csv"
		if os.path.exists(result_csv_file):
			# Read the existing CSV file
			existing_df = pd.read_csv(result_csv_file)
						
			# Find if this problem already exists in the CSV
			problem_index = existing_df[existing_df['name'] == to_problem.name].index
						
			if len(problem_index) > 0:
				# Update the existing entry
				existing_df.loc[problem_index[0]] = results_list[-1]
			else:
				# Append the new entry
				existing_df = pd.concat([existing_df, pd.DataFrame([results_list[-1]])], ignore_index=True)
						
			# Save the updated DataFrame back to CSV
			existing_df.to_csv(result_csv_file, index=False)

		else:
			# If the CSV file does not exist, create it with the new entry
			pd.DataFrame([results_list[-1]]).to_csv(result_csv_file, index=False)
		
		# Export the mesh with pseudo density to a .vtu file
		if (saveVTU) and (optimizationMethod == TO_METHODS.DENSITYMMA):
			vtu_dir= f"./Results/VTU"
			if not os.path.exists(vtu_dir):
				os.makedirs(vtu_dir)
			vtu_file = f"{vtu_dir}/{to_problem.name}.vtu"
			fe_solver.mesh.export_vtu_mesh(fe_solver.mesh.elemPseudoDensity,
										file_name = vtu_file,)
	# Convert results_list to a DataFrame for better visualization
	# Read the results from the existing CSV file if it exists, otherwise create a new DataFrame
	result_csv_file = f"{output_dir}/{optimizationMethod.name}_summary.csv"
	if os.path.exists(result_csv_file):
		results_df = pd.read_csv(result_csv_file)
	else:
		results_df = pd.DataFrame(results_list)
	if (results_df.empty):
		return
	# Format
	results_df['volfrac'] = results_df['volfrac'].map(lambda x: f"{x:.2g}")
	results_df['objective'] = results_df['objective'].map(lambda x: f"{x:.3g}")
	results_df['time (s)'] = results_df['time (s)'].map(lambda x: f"{x:.3g}")

	# Plot the results as a table
	fig, ax = plt.subplots(figsize=(10, len(results_list) * 0.5))
	ax.axis('tight')
	ax.axis('off')
	table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')
	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.auto_set_column_width(col=list(range(len(results_df.columns))))

	# Make the first row and column bold
	for key, cell in table.get_celld().items():
		if key[0] == 0 or key[1] == 0:  # Header row
			cell.set_text_props(weight='bold')
	
	# Save the table as an image
	results_path = f"{output_dir}/{optimizationMethod.name}_summary.png"

	plt.savefig(results_path, bbox_inches='tight')


def combine_results():
	# Get the latest results directory
	for subFolder in ["Structural-Compliance2.5D_1",
				   "Structural-Compliance2.5D_2",
				   "Structural-Compliance3D",
				   "Structural-NonCompliance", 
				   "Structural-BodyForce",
				   "ThermoStructural",
				   "Thermal-Compliance2.5D"]:
		# Get the latest results directory for the given subfolder
		# Use glob to find all matching directories and sort them
		# Use time.strftime to get the current date in the format YYYY-MM-DD
		# Sort the directories and take the last one (most recent)
		#results_dir = sorted(glob.glob(f"./Results/Results_{time.strftime('%Y-%m-%d')}/{subFolder}"))[-1]
		# If you want to combine results from all subfolders, uncomment the line below
		results_dirs = sorted(glob.glob(f"./Results/Results_{time.strftime('%Y-%m-%d')}/{subFolder}"))
		if not results_dirs:
			print(f"No results directory found for {subFolder}. Skipping...")
			continue
		results_dir = results_dirs[-1]
		print(f"Combining results from {results_dir}")
		# Read all CSV files
		dataframes = {}
		for method in TO_METHODS:
			csv_path = f"{results_dir}/{method.name}/{method.name}_summary.csv"
			if os.path.exists(csv_path):
				df = pd.read_csv(csv_path)
				# Convert compliance and time strings to float
				df['objective'] = df['objective'].astype(float)
				df['time (s)'] = df['time (s)'].astype(float)
				dataframes[method.name] = df

		if not dataframes:
			return

		# Create compliance summary
		problems = dataframes['DENSITYMMA']['name']
		compliance_data = {}
		fea_data = {}
		time_data = {}
		
		# Get reference values from DENSITYMMA
		reference_compliance = dict(zip(
			dataframes['DENSITYMMA']['name'],
			dataframes['DENSITYMMA']['objective']
		))
		reference_time = dict(zip(
			dataframes['DENSITYMMA']['name'],
			dataframes['DENSITYMMA']['time (s)']
		))

		# Calculate normalized compliance, time and get #FEAs for each method
		for method, df in dataframes.items():
			compliance_data[method] = [
				min(row['objective'] / reference_compliance[row['name']], 5)
				for _, row in df.iterrows()
			]
			time_data[method] = [
				row['time (s)'] / reference_time[row['name']]
				for _, row in df.iterrows()
			]
			# Get reference values for #FEAs from DENSITYMMA
			reference_feas = dict(zip(
				dataframes['DENSITYMMA']['name'],
				dataframes['DENSITYMMA']['#FEAs']
			))
			# Calculate normalized #FEAs
			fea_data[method] = [
				row['#FEAs'] / reference_feas[row['name']]
				for _, row in df.iterrows()
			]

		# Create and save normalized compliance summary
		compliance_df = pd.DataFrame(compliance_data, index=problems)
		#compliance_df.to_csv(f"{results_dir}/compliance_summary.csv")

		# Create and save normalized time summary
		time_df = pd.DataFrame(time_data, index=problems)
		#time_df.to_csv(f"{results_dir}/time_summary.csv")

		# Create and save #FEAs summary
		fea_df = pd.DataFrame(fea_data, index=problems)
		#fea_df.to_csv(f"{results_dir}/fea_summary.csv")
		# Create separate plots for compliance, time and FEAs
		
		# Plot normalized compliance
		plt.figure(figsize=(12, 6))
		compliance_df.plot(kind='bar', width=0.8)
		plt.title('Relative Compliance', fontsize=12, fontweight='bold')
		plt.ylabel('Relative to DENSITYMMA', fontsize=10)
		plt.xticks(rotation=45, fontsize=8, ha='right')
		plt.legend(title='Method', fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.grid(True, alpha=0.3)
		plt.subplots_adjust(right=0.85)  # Make room for legend
		plt.tight_layout()
		plt.savefig(f"{results_dir}/{subFolder}_compliance_comparison.png", dpi=300, bbox_inches='tight')
		plt.close()

		# Plot normalized time
		plt.figure(figsize=(10, 6))
		time_df.plot(kind='bar', width=0.8)
		plt.title('Relative Computation Time', fontsize=12, fontweight='bold') 
		plt.ylabel('Relative to DENSITYMMA', fontsize=10)
		plt.xticks(rotation=45, fontsize=8, ha='right')
		plt.legend(title='Method', fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(f"{results_dir}/{subFolder}_time_comparison.png", dpi=300, bbox_inches='tight')
		plt.close()

		# Plot number of FEAs
		plt.figure(figsize=(10, 6))
		fea_df.plot(kind='bar', width=0.8)
		plt.title('Relative Num. of FEAs', fontsize=12, fontweight='bold')
		plt.ylabel('Relative to DENSITYMMA', fontsize=10)
		plt.xticks(rotation=45, fontsize=8, ha='right')
		plt.legend(title='Method', fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(f"{results_dir}/{subFolder}_fea_comparison.png", dpi=300, bbox_inches='tight')
		plt.close()

		# Create and plot normalized volume fraction summary
		volume_data = {}
		for method, df in dataframes.items():
			volume_data[method] = [float(vol) for vol in df['volfrac']]
		
		volume_df = pd.DataFrame(volume_data, index=problems)
		
		plt.figure(figsize=(10, 6))
		volume_df.plot(kind='bar', width=0.8)
		plt.title('Volume Fraction', fontsize=12, fontweight='bold')
		plt.ylabel('Volume Fraction', fontsize=10)
		plt.xticks(rotation=45, fontsize=8, ha='right')
		plt.legend(title='Method', fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(f"{results_dir}/{subFolder}_volume_comparison.png", dpi=300, bbox_inches='tight')
		plt.close()

def create_summary_tables():
	subfolders = [
		"Structural-Compliance2.5D_1",
		"Structural-Compliance2.5D_2",
		"Structural-Compliance3D",
		"Structural-NonCompliance",
		"Structural-BodyForce",
		"ThermoStructural",
		"Thermal-Compliance2.5D",
	]

	for subFolder in subfolders:
		results_dirs = sorted(glob.glob(f"./Results/Results_{time.strftime('%Y-%m-%d')}/{subFolder}"))
		if not results_dirs:
			print(f"No results directory found for {subFolder}. Skipping...")
			continue

		results_dir = results_dirs[-1]
		method_dfs = {}

		for method in TO_METHODS:
			csv_path = f"{results_dir}/{method.name}/{method.name}_summary.csv"
			if os.path.exists(csv_path):
				df = pd.read_csv(csv_path)
				df = df[['name', 'volfrac', 'objective', 'time (s)']]
				df.rename(columns={
					'volfrac': f'{method.name}_volfrac',
					'objective': f'{method.name}_objective',
					'time (s)': f'{method.name}_time (s)'
				}, inplace=True)
				method_dfs[method.name] = df

		if not method_dfs:
			print(f"No method CSVs found in {results_dir}. Skipping...")
			continue

		# Start from the DENSITYMMA dataframe to get the problem list
		base_df = method_dfs.get('DENSITYMMA', next(iter(method_dfs.values()))).copy()
		for m, df in method_dfs.items():
			base_df = base_df.merge(df, on='name', how='outer')

		# Build rows and compute image paths per method
		rows = []
		methods = list(TO_METHODS)

		# Use the problem image saved in the Problems folder
		for _, row in base_df.iterrows():
			name = row['name']
			entry = {'Name': name, 'cells': {}}
			# Problem column uses the problem image generated in runTOMethodOnBenchmarks
			problem_img_path = f"{results_dir}/Problems/{name}_problem.png"
			entry['problem_img'] = problem_img_path if os.path.exists(problem_img_path) else None
			for method in methods:
				img_path = f"{results_dir}/{method.name}/{name}.png"
				entry['cells'][method.name] = {
					'img': img_path if os.path.exists(img_path) else None,
					'text': ""
				}
			rows.append(entry)

		# Create a grid figure with images per cell
		n_rows = len(rows)
		# Columns: Name | Problem | per-method images
		n_cols = 2 + len(methods)
		if n_rows == 0:
			print(f"No rows to plot for {results_dir}. Skipping...")
			continue

		cell_h = 1.8  # height per row in inches
		cell_w = 2.2  # width per image cell
		name_col_w = 2.8
		problem_col_w = 3.0
		fig_w = name_col_w + problem_col_w + len(methods) * cell_w
		fig_h = max(3, n_rows * cell_h)
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
		# Normalize axes to 2D array
		if n_rows == 1:
			axes = np.array([axes])
		if n_cols == 1:
			axes = axes.reshape(n_rows, 1)

		# Column headers in the first row's axes titles
		axes[0, 0].set_title("Name", fontsize=10, fontweight='bold')
		axes[0, 1].set_title("Problem", fontsize=10, fontweight='bold')
		for ci, method in enumerate(methods, start=2):
			axes[0, ci].set_title(method.name, fontsize=10, fontweight='bold')

		for ri, entry in enumerate(rows):
			# Name cell
			ax0 = axes[ri, 0]
			ax0.axis('off')
			ax0.text(0.02, 0.5, entry['Name'], fontsize=9, va='center', ha='left')

			# Problem image cell
			ax_prob = axes[ri, 1]
			ax_prob.axis('off')
			if entry['problem_img'] is not None:
				try:
					img = plt.imread(entry['problem_img'])
					ax_prob.imshow(img)
				except Exception:
					ax_prob.text(0.02, 0.5, "Problem image load error", fontsize=8, va='center', ha='left')
			else:
				ax_prob.text(0.02, 0.5, "No problem image", fontsize=8, va='center', ha='left')

			# Method cells: show figure if available, else text
			for ci, method in enumerate(methods, start=2):
				ax = axes[ri, ci]
				ax.axis('off')
				cell = entry['cells'][method.name]
				if cell['img'] is not None:
					try:
						img = plt.imread(cell['img'])
						ax.imshow(img)
						if cell['text']:
							ax.text(0.02, 0.02, cell['text'], fontsize=8, va='bottom', ha='left',
									color='white', bbox=dict(facecolor='black', alpha=0.4, pad=2),
									transform=ax.transAxes)
					except Exception:
						ax.text(0.02, 0.5, cell['text'] or "Image load error", fontsize=8, va='center', ha='left')
				else:
					ax.text(0.02, 0.5, cell['text'] or "No image", fontsize=8, va='center', ha='left')

		plt.tight_layout()
		out_path = f"{results_dir}/{subFolder}_summary_table.png"
		plt.savefig(out_path, dpi=300, bbox_inches='tight')
		plt.close()

if __name__ == "__main__":    
	
	optimizationMethods = [TO_METHODS.DENSITYMMA, TO_METHODS.DENSITYOCM,TO_METHODS.PARETO,TO_METHODS.LEVELSET]
	for optimizationMethod in optimizationMethods:
		runTOMethodOnBenchmarks(optimizationMethod)
		print("-" * 50)
		print(f"Finished {optimizationMethod.name} tests.")
		print("-" * 50)
		print("\n")
	
	# Combine results from all methods
	combine_results() 

	create_summary_tables()