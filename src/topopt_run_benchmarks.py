from topopt_common import *

from topopt_density_mma import topopt_mma
from topopt_density_oc import topopt_optimality_criteria	
from topopt_pareto import topopt_pareto
from topopt_levelset import topopt_levelset	
import glob

def runTOTests(optimizationMethod):

	# Create a list to store results
	results_list = []
	dsolver = deflation.DeflationSolver()


	benchmarks_problems = [StructuralTOExamples.MitchellOneLoad, StructuralTOExamples.MitchellTwoEqualLoads,
						StructuralTOExamples.MitchellTwoUnequalLoads, 
						StructuralTOExamples.MBBB,
						StructuralTOExamples.MidCantilever, StructuralTOExamples.EdgeCantilever, 
						StructuralTOExamples.ThreeHoleBracket, 
						 StructuralTOExamples.DistributedLoad, StructuralTOExamples.Multiload,
						 StructuralTOExamples.LBracket, StructuralTOExamples.TorquePlate,
						 StructuralTOExamples.KnuckleAssembly, StructuralTOExamples.Table]
	
	
	for to_problem in benchmarks_problems:
		print("-" * 50)
		print(f"Running {to_problem.name} using {optimizationMethod.name} method")
		print("-" * 50)

		mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)

		if (to_params.nDOFDesired < 100000):#  PARDISO 
			print("Solver: Pardiso")
			solver = lin_solv.Solvers.PARDISO
		else: 
			print("Solver: DPCG")
			solver = lin_solv.Solvers.DPCG 
			nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
			dsolver.create_deflation_groups(mesh, nGroups)
			dsolver.create_delfation_matrix(mesh)
			dsolver.W = dsolver.W[bc.free_dofs, :]

		
		fe_solver = fea.StructFEA(mesh = mesh,
					mat_prop = mat_prop,
					bc = bc,
					solver = solver,
					dsolver = dsolver,
					rtol = 1e-8,
					elem_body_force = elem_body_force)
		startTime = time.time()
		if optimizationMethod == TO_METHODS.DENSITYMMA:
			u, history,success,errorMsg,nFEAs = topopt_mma(fe_solver = fe_solver,
									to_params = to_params)
		elif optimizationMethod == TO_METHODS.DENSITYOC:
			u, history, success,errorMsg,nFEAs = topopt_optimality_criteria(fe_solver = fe_solver,
											to_params = to_params)
		elif optimizationMethod == TO_METHODS.PARETO:
			u, history, success,errorMsg,nFEAs = topopt_pareto(fe_solver = fe_solver,
													to_params = to_params)
		elif optimizationMethod == TO_METHODS.LEVELSET:
			u, history, success,errorMsg,nFEAs = topopt_levelset(fe_solver = fe_solver,
													to_params = to_params)
		timeTaken = time.time() - startTime
		# Create the directory if it does not exist
		output_dir = f"./Results/Results_{time.strftime('%Y-%m-%d')}/{optimizationMethod.name}"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		image_path = f"{output_dir}/{to_problem.name}.png"
		title = f"{optimizationMethod.name}: nDOF: {3*fe_solver.mesh.num_nodes}, vol: {history['volume'][-1]:0.2f}, J: {history['compliance'][-1]:.3g}, time: {timeTaken:.0f} s"
	
		plots.plotMesh(fe_solver.mesh, bc = None, u=None, save_path = image_path, title = title)
		
		results_list.append({
			'name': to_problem.name,
			'comment': to_params.Comment,  
			'ndof': 3*fe_solver.mesh.num_nodes,
			'volume': history['volume'][-1],
			'compliance': history['compliance'][-1],
			'#FEAs': nFEAs,
			'time (s)': timeTaken,
			'success': success,
			'error': errorMsg
		})

	# Convert results_list to a DataFrame for better visualization
	results_df = pd.DataFrame(results_list)

	# Format
	results_df['volume'] = results_df['volume'].map(lambda x: f"{x:.2g}")
	results_df['compliance'] = results_df['compliance'].map(lambda x: f"{x:.3g}")
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
	# Save the table as a CSV file
	csv_path = f"{output_dir}/{optimizationMethod.name}_summary.csv"
	results_df.to_csv(csv_path, index=False)
	plt.savefig(results_path, bbox_inches='tight')

def combine_results():
	# Get the latest results directory
	results_dir = sorted(glob.glob("./Results/Results_*"))[-1]
	
	# Read all CSV files
	dataframes = {}
	for method in TO_METHODS:
		csv_path = f"{results_dir}/{method.name}/{method.name}_summary.csv"
		if os.path.exists(csv_path):
			df = pd.read_csv(csv_path)
			# Convert compliance and time strings to float
			df['compliance'] = df['compliance'].astype(float)
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
		dataframes['DENSITYMMA']['compliance']
	))
	reference_time = dict(zip(
		dataframes['DENSITYMMA']['name'],
		dataframes['DENSITYMMA']['time (s)']
	))

	# Calculate normalized compliance, time and get #FEAs for each method
	for method, df in dataframes.items():
		compliance_data[method] = [
			row['compliance'] / reference_compliance[row['name']]
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

if __name__ == "__main__":    
	jax.config.update("jax_enable_x64", True)
	
	optimizationMethods = [TO_METHODS.DENSITYMMA, TO_METHODS.DENSITYOC, TO_METHODS.PARETO]
	optimizationMethods = [TO_METHODS.PARETO]
	for optimizationMethod in optimizationMethods:
		runTOTests(optimizationMethod)
		print(f"Finished {optimizationMethod.name} tests.")
		print("-" * 50)
		print("\n")
	
	# Combine results from all methods
	combine_results()