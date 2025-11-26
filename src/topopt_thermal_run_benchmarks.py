from topopt_common import *
from topopt_mma import topopt_mma
from topopt_ocm import topopt_optimality_criteria	
from topopt_pareto import topopt_pareto
from topopt_levelset import topopt_levelset	
from topopt_thermal_benchmarks import *
import time
import glob
import pandas as pd

def runTOMethodOnThermalBenchmarks(optimizationMethod):
	# Create a list to store results

	results_list = []
	dsolver = deflation.DeflationSolver()
 
	feaMode = FEA_MODE.THERMAL

	benchmarks_2_5D_problems = [ThermalTOExamples.HeatPlate, ThermalTOExamples.FourCornersThermal,
							 ThermalTOExamples.BridgeThermal]	
	
	for to_problem in benchmarks_2_5D_problems:
		if to_problem in benchmarks_2_5D_problems:
			subFolder = "Compliance2.5D"
		else:
			subFolder = "Compliance3D"
		print("-" * 50)
		print(f"Running {to_problem.name} using {optimizationMethod.name} method")
		print("-" * 50)
		print_progress = False
		mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)
		dsolver = deflation.DeflationSolver()
		if (to_params.nDOFDesired <= DIRECT_SOLVER_DOF_CUTOFF):#  # Choose solver. Typically PARDISO, but DPCG for large DOF problems
			solver = lin_solv.Solvers.PARDISO
		else:
			solver = lin_solv.Solvers.DPCG
			nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
			dsolver.create_deflation_groups(mesh, nGroups)
			dsolver.create_deflation_matrix(mesh)
			dsolver.W = dsolver.W[bc.free_dofs, :]

		startTime = time.time()
		
		fe_solver = hex_thermal_fea.HexThermalFEA(mesh = mesh,
					mat_prop = mat_prop,
					bc = bc,
					solver = solver,
					dsolver = dsolver,
					rtol = 1e-8,
					elem_body_force = elem_body_force)
		
		if optimizationMethod == TO_METHODS.DENSITYMMA:
			u, history,success,errorMsg,nFEAs = topopt_mma(feaMode,None,fe_solver,
									to_params = to_params,print_progress = print_progress)
		elif optimizationMethod == TO_METHODS.DENSITYOCM:
			u, history, success,errorMsg,nFEAs = topopt_optimality_criteria(feaMode, fe_solver,
											to_params = to_params,print_progress = print_progress)
		elif optimizationMethod == TO_METHODS.PARETO:
			u, history, success,errorMsg,nFEAs = topopt_pareto(feaMode, fe_solver,
													to_params = to_params,print_progress = print_progress)
		elif optimizationMethod == TO_METHODS.LEVELSET:
			u, history, success,errorMsg,nFEAs = topopt_levelset(feaMode, fe_solver,
													to_params = to_params,print_progress = print_progress)
		timeTaken = time.time() - startTime
		# Create the directory if it does not exist
		output_dir = f"./Results/Results_{time.strftime('%Y-%m-%d')}/Thermal/{subFolder}/{optimizationMethod.name}"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		image_path = f"{output_dir}/{to_problem.name}.png"
		title = f"{optimizationMethod.name}:  vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"
	
		fe_solver.plot_mesh(save_path=image_path, plot_bc = None, title=title)
		
		results_list.append({
			'name': to_problem.name,
			'comment': to_params.Comment,  
			'ndof': fe_solver.mesh.num_nodes,
			'volfrac': history['volfrac'][-1],
			'objective': history['objective'][-1],
			'#FEAs': nFEAs,
			'time (s)': timeTaken,
			'success': success,
			'error': errorMsg
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
		
	
	# Convert results_list to a DataFrame for better visualization

	# Read the results from the existing CSV file if it exists, otherwise create a new DataFrame
	result_csv_file = f"{output_dir}/{optimizationMethod.name}_summary.csv"
	if os.path.exists(result_csv_file):
		results_df = pd.read_csv(result_csv_file)
	else:
		results_df = pd.DataFrame(results_list)

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
	for subFolder in ["Compliance2.5D", "Compliance3D"]:
		# Get the latest results directory for the given subfolder
		# Use glob to find all matching directories and sort them
		# Use time.strftime to get the current date in the format YYYY-MM-DD
		# Sort the directories and take the last one (most recent)
		#results_dir = sorted(glob.glob(f"./Results/Results_{time.strftime('%Y-%m-%d')}/{subFolder}"))[-1]
		# If you want to combine results from all subfolders, uncomment the line below
		results_dirs = sorted(glob.glob(f"./Results/Results_{time.strftime('%Y-%m-%d')}/Thermal/{subFolder}"))
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
				row['objective'] / reference_compliance[row['name']]
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
		plt.savefig(f"{results_dir}/compliance_comparison.png", dpi=300, bbox_inches='tight')
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
		plt.savefig(f"{results_dir}/time_comparison.png", dpi=300, bbox_inches='tight')
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
		plt.savefig(f"{results_dir}/fea_comparison.png", dpi=300, bbox_inches='tight')
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
		plt.savefig(f"{results_dir}/volume_comparison.png", dpi=300, bbox_inches='tight')
		plt.close()

if __name__ == "__main__":    
	
	optimizationMethods = [TO_METHODS.DENSITYMMA, TO_METHODS.DENSITYOCM, TO_METHODS.PARETO]
	for optimizationMethod in optimizationMethods:
		runTOMethodOnThermalBenchmarks(optimizationMethod)
		print(f"Finished {optimizationMethod.name} tests.")
		print("-" * 50)
		print("\n")
	
	# Combine results from all methods
	combine_results()   