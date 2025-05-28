# Core scientific and optimization libraries
"""EvoMapX: Advanced Evolutionary Algorithm Framework

Author: Bilal H. Abed-alguni
Email: Bilal.h@yu.edu.jo
Affiliation: Department of Computer Sciences, Yarmouk University, Irbid, Jordan
"""

# Core scientific and optimization libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import uuid
import random
from opfunu.cec_based import cec2021
import inspect
from collections import defaultdict
import sys
import datetime
from fpdf import FPDF
import seaborn as sns
import matplotlib.pyplot as plt

# Configure matplotlib for high-quality publication figures
# Set seaborn style for clean, professional look
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '-'})
sns.set_context("paper", rc={"lines.linewidth": 2.5})

# Update matplotlib parameters for publication quality
plt.rcParams.update({
    # High-resolution output
    'figure.dpi': 300,
    'savefig.dpi': 300,
    
    # Font settings
    'font.size': 14,
    'font.family': 'serif',
    'font.weight': 'bold',
    'font.serif': ['Times New Roman'],
    
    # Axes settings
    'axes.linewidth': 2.5,
    'axes.labelweight': 'bold',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.edgecolor': 'black',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    
    # Grid settings
    'grid.linewidth': 1.0,
    'grid.alpha': 0.5,
    
    # Line and marker properties
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
    'lines.markeredgewidth': 2.0,
    
    # Legend properties
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'black',
    'legend.borderpad': 0.4,
    
    # Figure layout
    'figure.figsize': [8, 6],
    'figure.constrained_layout.use': True,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # Tick parameters
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# ==== Optimization Algorithm Parameters ====
DIM = 10          # Problem dimension
POP_SIZE = 20     # Population size
MAX_GEN = 100     # Maximum generations
ESC_TOLERANCE = 15 # Early stopping criterion tolerance
BOUNDS = (-100, 100) # Search space bounds
F = 0.5           # DE scaling factor
CR = 0.9          # DE crossover rate



# Create a timestamp for unique filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
txt_filename = f"output_{timestamp}.txt"
pdf_filename = f"output_{timestamp}.pdf"

# Redirect stdout to both console and text file
original_stdout = sys.stdout
text_file = open(txt_filename, 'w', encoding='utf-8')

class TeeOutput:
    def write(self, text):
        original_stdout.write(text)
        text_file.write(text)
    def flush(self):
        original_stdout.flush()
        text_file.flush()

sys.stdout = TeeOutput()

try:
    
    def validate_cec2021_function_output(func, func_name="Unknown Function", dim=30, bounds=(-100, 100), samples=1000):
        """Sample the selected CEC2021 function to verify output range."""
        outputs = []
        for _ in range(samples):
            x = np.random.uniform(bounds[0], bounds[1], dim)
            y = func(x)
            outputs.append(y)
    
        min_val = np.min(outputs)
        max_val = np.max(outputs)
        mean_val = np.mean(outputs)
    
        print("\n📊 Function Output Validation:")
        print(f"  Function: {func_name}")
        print(f"  Min:  {min_val:.4f}")
        print(f"  Max:  {max_val:.4f}")
        print(f"  Mean: {mean_val:.4f}")
        print("  🔍 Global optimum (from CEC2021 spec) should be near the min value.")
    
    
    # ==== User Selection of CEC2021 Function ====
    cec2021_funcs = {name: cls for name, cls in inspect.getmembers(cec2021, inspect.isclass) if name.startswith("F")}
    print("Available CEC2021 functions:")
    for idx, name in enumerate(cec2021_funcs.keys(), 1):
        print(f"{idx}: {name}")
    
    while True:
        try:
            choice = int(input(f"\nSelect a CEC2021 function by number (1-{len(cec2021_funcs)}): "))
            if 1 <= choice <= len(cec2021_funcs):
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a valid integer.")
    
    selected_func_name = list(cec2021_funcs.keys())[choice-1]
    print(f"Selected function: {selected_func_name}")
    selected_func_class = cec2021_funcs[selected_func_name]
    cec_f = selected_func_class(ndim=DIM)
    
    
    
    
    def cec2021_f(x):
        return cec_f.evaluate(x)
    
    
    # Validate function output range
    validate_cec2021_function_output(cec2021_f, func_name=selected_func_name, dim=DIM, bounds=BOUNDS, samples=1000)
    
    
    
    # ==== EvoMapX Structures ====
    class EvoIndividual:
        # Individual in population with tracking for evolutionary lineage
        def __init__(self, vector, fitness, algo, parent_ids=None, operator=None):
            self.id = str(uuid.uuid4())      # Unique identifier
            self.vector = vector             # Solution vector
            self.fitness = fitness           # Fitness value
            self.algo = algo                 # Algorithm that created this individual
            self.parent_ids = parent_ids or [] # Parent individuals' IDs
            self.operator = operator         # Operator that created this individual
    
    # ==== OAM (Operator Attribution Matrix) and PEG (Population Evolution Graph) Initialization ====
OAM = {
        "GA": {"crossover": [[]], "mutation": [[]]},              # Genetic Algorithm operators
        "PSO": {"velocity_update": [[]]},                        # Particle Swarm Optimization
        "CS": {                                                  # Cuckoo Search operators
        "levy_flight": [[]],
        "successful_replacement": [[]],
        "unsuccessful_attempt": [[]],
        "abandoned_nest": [[]],
        "random_init": [[]]},
        "DE": {"mutation": [[]], "crossover": [[]]}             # Differential Evolution operators
    }
    
    # ==== CDS Analysis Functions ====
    def calculate_operator_cds(oam_data, operator_name, algo_peg=None, generation_map=None, decay_factor=0.9):
        # Calculates Contribution Degree Sequence with temporal decay and lineage tracking
        """Calculate Contribution Degree Sequence for a specific operator with lineage-based influence.
        
        Args:
            oam_data: List of lists containing improvement values per generation
            operator_name: Name of the operator being analyzed
            algo_peg: Algorithm-specific PEG for lineage tracking
            generation_map: Dictionary mapping node IDs to generation numbers
            decay_factor: Factor for decaying influence over generations (default: 0.9)
        """
        if not oam_data:
            return []
        
        # Initialize CDS array
        cds = [0.0] * len(oam_data)
        
        # Process each generation's direct improvements
        for gen, gen_data in enumerate(oam_data):
            if not gen_data:
                continue
                
            # Calculate base contribution for current generation
            base_contribution = sum(max(0, imp) for imp in gen_data)
            cds[gen] += base_contribution
            
            # If we have PEG data, calculate delayed impacts
            if algo_peg and generation_map:
                # Find nodes from this generation
                gen_nodes = [node for node, g in generation_map.items() if g == gen]
                
                for node in gen_nodes:
                    if algo_peg.nodes[node].get('op') == operator_name:
                        # Get all descendants
                        descendants = nx.descendants(algo_peg, node)
                        
                        for desc in descendants:
                            if desc in generation_map:
                                desc_gen = generation_map[desc]
                                if desc_gen > gen:
                                    # Calculate delayed impact with decay
                                    gen_diff = desc_gen - gen
                                    impact = base_contribution * (decay_factor ** gen_diff)
                                    cds[desc_gen] += impact
        
        return cds

    def analyze_algorithm_operators(algo_oam, algo_name=None):
        """Analyze the effectiveness of operators for a specific algorithm with lineage tracking.
        
        Args:
            algo_oam: Algorithm-specific OAM data
            algo_name: Name of the algorithm for PEG tracking
        """
        # Comprehensive analysis of operator performance with lineage-based metrics
        operator_stats = {}
        
        # Build generation mapping if we have PEG data
        generation_map = {}
        if algo_name and algo_name in PEGs:
            algo_peg = PEGs[algo_name]
            
            # Map nodes to generations based on longest path from roots
            roots = [n for n in algo_peg.nodes() if algo_peg.in_degree(n) == 0]
            for node in algo_peg.nodes():
                max_path_length = 0
                for root in roots:
                    try:
                        path_length = len(nx.shortest_path(algo_peg, root, node)) - 1
                        max_path_length = max(max_path_length, path_length)
                    except nx.NetworkXNoPath:
                        continue
                generation_map[node] = max_path_length
        
        for operator, data in algo_oam.items():
            # Calculate CDS with lineage tracking
            cds = calculate_operator_cds(
                data, 
                operator,
                algo_peg=PEGs.get(algo_name) if algo_name else None,
                generation_map=generation_map if generation_map else None
            )
            
            if not cds:
                continue
                
            # Calculate enhanced statistics
            total_contribution = sum(cds)
            avg_contribution = total_contribution / len(cds) if cds else 0
            max_contribution = max(cds) if cds else 0
            active_gens = sum(1 for x in cds if x > 0)
            activity_rate = active_gens / len(cds) if cds else 0
            
            # Calculate influence persistence
            non_zero_indices = [i for i, x in enumerate(cds) if x > 0]
            influence_span = max(non_zero_indices) - min(non_zero_indices) + 1 if non_zero_indices else 0
            influence_density = len(non_zero_indices) / influence_span if influence_span > 0 else 0
            
            operator_stats[operator] = {
                "total_contribution": total_contribution,
                "avg_contribution": avg_contribution,
                "max_contribution": max_contribution,
                "activity_rate": activity_rate,
                "influence_span": influence_span,
                "influence_density": influence_density
            }
        
        return operator_stats

    def print_operator_analysis():
        """Print comprehensive analysis of operator effectiveness across all algorithms with lineage tracking."""
        # Generates detailed performance reports for all operators across algorithms
        print("\n📊 Enhanced Operator Effectiveness Analysis")
        
        for algo, operators in OAM.items():
            print(f"\n{algo} Algorithm:")
            stats = analyze_algorithm_operators(operators, algo_name=algo)
            
            # Sort operators by total contribution
            sorted_ops = sorted(stats.items(), 
                              key=lambda x: x[1]["total_contribution"], 
                              reverse=True)
            
            for op_name, op_stats in sorted_ops:
                print(f"\n  {op_name}:")
                print(f"    Direct Contribution:")
                print(f"      Total: {op_stats['total_contribution']:.4f}")
                print(f"      Average: {op_stats['avg_contribution']:.4f}")
                print(f"      Maximum: {op_stats['max_contribution']:.4f}")
                print(f"    Influence Analysis:")
                print(f"      Activity Rate: {op_stats['activity_rate']*100:.1f}%")
                print(f"      Influence Span: {op_stats['influence_span']} generations")
                print(f"      Influence Density: {op_stats['influence_density']*100:.1f}%")
                
                # Calculate PEG-based metrics
                if algo in PEGs:
                    algo_peg = PEGs[algo]
                    op_nodes = [n for n, d in algo_peg.nodes(data=True) 
                               if d.get('op') == op_name]
                    if op_nodes:
                        avg_descendants = sum(len(nx.descendants(algo_peg, n)) 
                                            for n in op_nodes) / len(op_nodes)
                        print(f"    Lineage Impact:")
                        print(f"      Average Descendants: {avg_descendants:.2f}")
    PEG = nx.DiGraph()
    PEGs = {
        "GA": nx.DiGraph(),
        "PSO": nx.DiGraph(),
        "CS": nx.DiGraph(),
        "DE": nx.DiGraph()
    }
    
    def log_to_peg(child, parents):
        # Unified PEG
        PEG.add_node(child.id, algo=child.algo, op=child.operator)
        for p in parents:
            PEG.add_edge(p.id, child.id)
    
        # Individual PEG for the algorithm
        algo_peg = PEGs[child.algo]
        algo_peg.add_node(child.id, algo=child.algo, op=child.operator)
        for p in parents:
            algo_peg.add_edge(p.id, child.id)
    
    # ==== Shared Population Initialization ====
    def initialize_population(algo_name):
        """Initialize population using Latin Hypercube Sampling for consistent initialization across algorithms."""
        # Uses LHS for better initial population diversity and consistent comparison
        population = []
        samples = np.random.random((POP_SIZE, DIM))
        
        # Latin Hypercube Sampling
        for i in range(POP_SIZE):
            for j in range(DIM):
                samples[i, j] = (samples[i, j] + i) / POP_SIZE
        
        # Create individuals with proper initialization
        for i in range(POP_SIZE):
            vector = BOUNDS[0] + samples[i] * (BOUNDS[1] - BOUNDS[0])
            ind = EvoIndividual(vector, 0, algo_name)
            ind.fitness = cec2021_f(ind.vector)
            population.append(ind)
            PEG.add_node(ind.id, algo=algo_name)
        
        return population
    
    # ==== GA ====
    def calculate_diversity(population):
        """Calculate population diversity using mean pairwise Euclidean distance."""
        # Measures population diversity for adaptive control
        n = len(population)
        if n <= 1:
            return 0.0
        
        total_distance = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_distance += np.linalg.norm(population[i].vector - population[j].vector)
                count += 1
        return total_distance / count if count > 0 else 0.0

    def adaptive_tournament_size(diversity, min_k=2, max_k=5):
        """Adapt tournament size based on population diversity."""
        # Dynamically adjusts selection pressure based on population diversity
        # Higher diversity -> smaller tournament size (less selection pressure)
        # Lower diversity -> larger tournament size (more selection pressure)
        normalized_diversity = np.clip(diversity / (BOUNDS[1] - BOUNDS[0]), 0, 1)
        k = min_k + (1 - normalized_diversity) * (max_k - min_k)
        return int(round(k))

    def ga_run():
        # Genetic Algorithm with adaptive operators and diversity management
        # Use shared initialization
        population = initialize_population("GA")
    
        # Initialize fitness tracking
        best_fitness = min(population, key=lambda ind: ind.fitness).fitness
        fitness_curve = [best_fitness]
        no_improve = 0
        diversity_history = []
        stagnation_counter = 0
    
        for gen in range(MAX_GEN):
            OAM["GA"]["crossover"].append([])
            OAM["GA"]["mutation"].append([])
    
            # Monitor population diversity for adaptive control
            current_diversity = calculate_diversity(population)
            diversity_history.append(current_diversity)
            
            # Dynamically adjust selection pressure through tournament size
            tournament_k = adaptive_tournament_size(current_diversity)
            
            new_pop = []
            # Preserve top 10% as elites (increased from just the best individual)
            elites = sorted(population, key=lambda ind: ind.fitness)[:max(1, int(POP_SIZE * 0.1))]
            new_pop.extend(elites)
            
            # Tournament selection with adaptive size
            def tournament_select(pop):
                selected = random.sample(pop, tournament_k)
                return min(selected, key=lambda ind: ind.fitness)
            
            # Adapt mutation rate based on diversity and generation progress
            base_mutation_rate = 0.1 * (1 - gen/MAX_GEN) + 0.01  # Decreases over time
            diversity_factor = np.clip(1.0 - current_diversity / (BOUNDS[1] - BOUNDS[0]), 0.5, 2.0)  # Increases when diversity is low
            mutation_rate = base_mutation_rate * diversity_factor  # Combined adaptive rate
            
            while len(new_pop) < POP_SIZE:
                # Tournament selection
                p1 = tournament_select(population)
                p2 = tournament_select(population)
                
                # Adaptive crossover based on parent similarity
                parent_distance = np.linalg.norm(p1.vector - p2.vector)
                parent_similarity = 1.0 - parent_distance / (BOUNDS[1] - BOUNDS[0])
                
                # Two-point crossover with adaptive mixing
                points = sorted(np.random.choice(range(1, DIM), 2, replace=False))
                mix_ratio = np.random.uniform(0.3, 0.7) if parent_similarity > 0.8 else 0.5
                child_vec = np.concatenate((
                    p1.vector[:points[0]],
                    mix_ratio * p1.vector[points[0]:points[1]] + (1 - mix_ratio) * p2.vector[points[0]:points[1]],
                    p1.vector[points[1]:]
                ))
                
                pre_mut = child_vec.copy()
                
                # Adaptive mutation strength based on diversity and fitness improvement
                base_strength = 0.3 * (1 - gen/MAX_GEN) + 0.05
                mutation_strength = base_strength * diversity_factor
                
                # Non-uniform mutation with adaptive rate
                if np.random.rand() < 0.8:  # 80% chance to apply mutation
                    mutation_mask = np.random.rand(DIM) < mutation_rate
                    child_vec[mutation_mask] += np.random.normal(0, mutation_strength, size=sum(mutation_mask))
                
                child_vec = np.clip(child_vec, *BOUNDS)
                pre_mut_fit = cec2021_f(pre_mut)
                child_fit = cec2021_f(child_vec)
    
                OAM["GA"]["crossover"][-1].append(max(p1.fitness, p2.fitness) - pre_mut_fit)
                OAM["GA"]["mutation"][-1].append(pre_mut_fit - child_fit)
    
                child = EvoIndividual(child_vec, child_fit, "GA", [p1.id, p2.id], "crossover+mutation")
                log_to_peg(child, [p1, p2])
                new_pop.append(child)
    
            population = sorted(new_pop, key=lambda ind: ind.fitness)[:POP_SIZE]
            current_best = population[0]
            fitness_curve.append(current_best.fitness)
    
            # Track best fitness and update counters
            if current_best.fitness < best_fitness - 1e-6:
                best_fitness = current_best.fitness
                no_improve = 0
                stagnation_counter = 0
            else:
                no_improve += 1
                stagnation_counter += 1
            
            # Apply diversity maintenance without early termination
            if stagnation_counter >= ESC_TOLERANCE:
                # Reinitialize population with varying degrees of perturbation
                for i in range(1, POP_SIZE):
                    if i < POP_SIZE // 3:  # Small perturbation
                        perturbation = np.random.normal(0, 0.1 * (BOUNDS[1] - BOUNDS[0]), DIM)
                        new_vec = np.clip(population[i].vector + perturbation, *BOUNDS)
                    elif i < 2 * POP_SIZE // 3:  # Medium perturbation
                        perturbation = np.random.normal(0, 0.3 * (BOUNDS[1] - BOUNDS[0]), DIM)
                        new_vec = np.clip(population[i].vector + perturbation, *BOUNDS)
                    else:  # Complete reinitialization
                        new_vec = np.random.uniform(*BOUNDS, DIM)
                    population[i].vector = new_vec
                    population[i].fitness = cec2021_f(new_vec)
                stagnation_counter = 0
    
        return fitness_curve
    
    # ==== PSO ====
    def pso_run():
        # Particle Swarm Optimization with adaptive parameters and component tracking
        # Use linearly decreasing inertia weight for exploration-exploitation balance
        w_start, w_end = 0.9, 0.4
        
        # Use time-varying acceleration coefficients for dynamic search behavior
        c1_start, c1_end = 2.5, 0.5  # Cognitive component decreases over time
        c2_start, c2_end = 0.5, 2.5  # Social component increases over time
        
        # Initialize OAM structure
        OAM["PSO"] = {
            "cognitive_component": [[]],
            "social_component": [[]],
            "inertia_component": [[]],
            "position_update": [[]]
        }
        
        # Use shared initialization
        particles = initialize_population("PSO")
        
        # Initialize velocities and personal bests
        v_max = 0.2 * (BOUNDS[1] - BOUNDS[0])  # Limit velocity
        velocities = [np.random.uniform(-v_max, v_max, DIM) for _ in range(POP_SIZE)]
        personal_bests = [{"position": p.vector.copy(), "fitness": p.fitness} for p in particles]
        global_best = min(personal_bests, key=lambda x: x["fitness"])
    
        fitness_curve = [global_best["fitness"]]
        no_improve = 0
        stagnation_counter = 0
    
        for gen in range(MAX_GEN):
            # Dynamic parameters
            w = w_start - (w_start - w_end) * gen / MAX_GEN
            c1 = c1_start - (c1_start - c1_end) * gen / MAX_GEN
            c2 = c2_start + (c2_end - c2_start) * gen / MAX_GEN
            
            # Initialize OAM entries
            for op in OAM["PSO"]:
                OAM["PSO"][op].append([])
            
            current_gen_best = {"fitness": float('inf')}
            
            for i in range(POP_SIZE):
                original_position = particles[i].vector.copy()
                original_fitness = particles[i].fitness
                
                # Calculate inertia component
                inertia = w * velocities[i]
                position_after_inertia = np.clip(original_position + inertia, *BOUNDS)
                fitness_after_inertia = cec2021_f(position_after_inertia)
                inertia_improvement = original_fitness - fitness_after_inertia
                OAM["PSO"]["inertia_component"][-1].append(inertia_improvement)
                
                # Calculate cognitive component with random vector
                r1 = np.random.rand(DIM)
                cognitive = c1 * r1 * (personal_bests[i]["position"] - original_position)
                position_after_cognitive = np.clip(position_after_inertia + cognitive, *BOUNDS)
                fitness_after_cognitive = cec2021_f(position_after_cognitive)
                cognitive_improvement = fitness_after_inertia - fitness_after_cognitive
                OAM["PSO"]["cognitive_component"][-1].append(cognitive_improvement)
                
                # Calculate social component with random vector
                r2 = np.random.rand(DIM)
                social = c2 * r2 * (global_best["position"] - original_position)
                position_after_social = np.clip(position_after_cognitive + social, *BOUNDS)
                fitness_after_social = cec2021_f(position_after_social)
                social_improvement = fitness_after_cognitive - fitness_after_social
                OAM["PSO"]["social_component"][-1].append(social_improvement)
                
                # Update velocity with limited magnitude
                velocities[i] = inertia + cognitive + social
                
                # Velocity clamping
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                # Apply velocity constriction if needed
                if np.random.rand() < 0.2:  # 20% chance for constriction
                    velocities[i] *= 0.729  # Constriction factor
                
                new_position = np.clip(original_position + velocities[i], *BOUNDS)
                new_fitness = cec2021_f(new_position)
                
                # Calculate improvement
                total_improvement = original_fitness - new_fitness
                OAM["PSO"]["position_update"][-1].append(total_improvement)
                
                component_improvements = {
                    "inertia": inertia_improvement,
                    "cognitive": cognitive_improvement,
                    "social": social_improvement
                }
                dominant_component = max(component_improvements, key=component_improvements.get)
                operator_name = f"pso_{dominant_component}"
                
                new_particle = EvoIndividual(new_position, new_fitness, "PSO", 
                                           [particles[i].id], operator_name)
                log_to_peg(new_particle, [particles[i]])
                
                # Update personal best
                if new_fitness < personal_bests[i]["fitness"]:
                    personal_bests[i] = {"position": new_position.copy(), "fitness": new_fitness}
                
                # Track generation's best
                if new_fitness < current_gen_best["fitness"]:
                    current_gen_best = {"position": new_position.copy(), "fitness": new_fitness}
                
                particles[i] = new_particle
            
            # Update global best
            if current_gen_best["fitness"] < global_best["fitness"]:
                global_best = current_gen_best.copy()
                no_improve = 0
                stagnation_counter = 0
            else:
                no_improve += 1
                stagnation_counter += 1
            
            # Add perturbation when stagnation occurs without early termination
            if stagnation_counter >= 5:  # If no improvement for 5 generations
                # Perturb 30% of particles except the best one
                for i in range(1, int(POP_SIZE * 0.3)):
                    idx = np.random.randint(POP_SIZE)
                    perturbation = np.random.normal(0, 0.1 * (BOUNDS[1] - BOUNDS[0]), DIM)
                    particles[idx].vector = np.clip(particles[idx].vector + perturbation, *BOUNDS)
                    particles[idx].fitness = cec2021_f(particles[idx].vector)
                stagnation_counter = 0
            
            fitness_curve.append(global_best["fitness"])
    
        return fitness_curve
    
    #==== CS ====
    def levy_flight(Lambda=1.5, step_size=0.1, gen=0, max_gen=MAX_GEN):
        # Generates Lévy flight step sizes with adaptive scaling over generations
        # Adaptive step size that decreases with generations
        adaptive_step = step_size * (1 - 0.9 * gen / max_gen)
        
        # Use math module instead of np.math
        sigma = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
                 (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
        u = np.random.normal(0, sigma, DIM)
        v = np.random.normal(0, 1, DIM)
        return adaptive_step * (u / np.abs(v) ** (1 / Lambda))
    
    def cs_run():
        # Cuckoo Search with adaptive abandonment and elite preservation
        # Use shared initialization
        nests = initialize_population("CS")
        best = min(nests, key=lambda n: n.fitness)
        best_fitness = best.fitness
        fitness_curve = [best_fitness]
        no_improve = 0
        
        # Dynamic abandonment rate for exploration-exploitation balance
        pa_start, pa_end = 0.5, 0.1  # Higher at start for exploration, lower at end for exploitation
        
        # Elitism strategy to preserve best solutions
        elite_pct = 0.1
        
        for gen in range(MAX_GEN):
            # Adaptive abandonment probability
            pa = pa_start - (pa_start - pa_end) * gen / MAX_GEN
            
            # Initialize OAM arrays
            for op in OAM["CS"]:
                OAM["CS"][op].append([])
            
            # Sort nests by fitness
            nests = sorted(nests, key=lambda n: n.fitness)
            elite_count = max(1, int(POP_SIZE * elite_pct))
            elite_nests = nests[:elite_count]
            remaining_nests = nests[elite_count:]
            
            # Process non-elite nests
            new_nests = list(elite_nests)  # Start with elites
            
            for i, nest in enumerate(remaining_nests):
                # Adaptive Lévy flight step size based on nest quality
                quality_factor = 1.0 - i / len(remaining_nests)  # Better nests get smaller steps
                
                # Generate two different step types with different probabilities
                if np.random.random() < 0.7:  # 70% standard Lévy flights
                    step = levy_flight(step_size=0.2 * quality_factor, gen=gen, max_gen=MAX_GEN)
                    # Mix step with best solution for better exploitation
                    new_vec = nest.vector + step * (nest.vector - best.vector)
                else:  # 30% more exploitative step towards best
                    # Direct movement towards best solution with random scaling
                    step_scale = np.random.random() * 0.5  # Scale factor between 0 and 0.5
                    new_vec = nest.vector + step_scale * (best.vector - nest.vector)
                
                new_vec = np.clip(new_vec, *BOUNDS)
                fit = cec2021_f(new_vec)
                
                delta = nest.fitness - fit
                OAM["CS"]["levy_flight"][-1].append(delta)
                child = EvoIndividual(new_vec, fit, "CS", [nest.id], "levy_flight")
                log_to_peg(child, [nest])
    
                if fit < nest.fitness:
                    OAM["CS"]["successful_replacement"][-1].append(delta)
                    new_nests.append(child)
                else:
                    OAM["CS"]["unsuccessful_attempt"][-1].append(0)
                    new_nests.append(nest)
            
            # Process abandonment - avoid abandoning elite nests
            nests = sorted(new_nests, key=lambda n: n.fitness)
            
            # Skip elite nests for abandonment
            for i in range(elite_count, POP_SIZE):
                # Higher abandonment probability for worse nests
                abandon_prob = pa * (1 + (i - elite_count) / (POP_SIZE - elite_count))
                abandon_prob = min(0.95, abandon_prob)  # Cap at 95%
                
                if random.random() < abandon_prob:
                    # Generate new nest with some bias towards the best solution
                    if random.random() < 0.3:  # 30% chance to use best solution as base
                        # Create nest around best solution
                        new_vec = best.vector + np.random.normal(0, 0.1 * (BOUNDS[1] - BOUNDS[0]), DIM)
                    else:
                        # Create completely new solution
                        new_vec = np.random.uniform(*BOUNDS, DIM)
                    
                    fit = cec2021_f(new_vec)
                    OAM["CS"]["abandoned_nest"][-1].append(nests[i].fitness)
                    OAM["CS"]["random_init"][-1].append(fit)
    
                    abandoned = EvoIndividual(new_vec, fit, "CS", operator="abandoned")
                    PEG.add_node(abandoned.id, algo="CS", op="abandoned")
                    nests[i] = abandoned
            
            current_best = min(nests, key=lambda n: n.fitness)
            fitness_curve.append(current_best.fitness)
            
            # Update best solution
            if current_best.fitness < best.fitness:
                best = current_best
            
            # Check improvement and maintain diversity without early termination
            if current_best.fitness < best_fitness - 1e-6:
                best_fitness = current_best.fitness
                no_improve = 0
            else:
                no_improve += 1
                
                # Add periodic restart mechanism without breaking the loop
                if no_improve >= ESC_TOLERANCE // 2:
                    # Reinitialize 50% of non-elite population
                    for i in range(elite_count, POP_SIZE):
                        if random.random() < 0.5:
                            new_vec = np.random.uniform(*BOUNDS, DIM)
                            fit = cec2021_f(new_vec)
                            nests[i] = EvoIndividual(new_vec, fit, "CS", operator="restart")
                            PEG.add_node(nests[i].id, algo="CS", op="restart")
                    no_improve = 0  # Reset counter after restart
    
        return fitness_curve
    
    # ==== DE ====
    def de_run():
        # Differential Evolution with success-based parameter adaptation
        # Use shared initialization
        population = initialize_population("DE")
        best_fitness = min(population, key=lambda x: x.fitness).fitness
        fitness_curve = [best_fitness]
        no_improve = 0
        
        # Parameter adaptation ranges
        F_min, F_max = 0.2, 1.0    # Mutation scale factor bounds
        CR_min, CR_max = 0.5, 0.9  # Crossover rate bounds
        
        # Dictionary to track success rate of parameter values
        success_history = {
            "F": [],
            "CR": []
        }
    
        for gen in range(MAX_GEN):
            OAM["DE"]["mutation"].append([])
            OAM["DE"]["crossover"].append([])
            
            # Adapt parameters based on success history
            if len(success_history["F"]) > 0:
                mean_F = np.mean(success_history["F"][-10:]) if success_history["F"] else F
                mean_CR = np.mean(success_history["CR"][-10:]) if success_history["CR"] else CR
            else:
                mean_F = F
                mean_CR = CR
            
            new_pop = []
            successful_F = []
            successful_CR = []
            
            sorted_pop = sorted(population, key=lambda x: x.fitness)
            
            for i in range(POP_SIZE):
                # Adaptive F and CR values
                current_F = mean_F + 0.1 * np.random.normal(0, 1)
                current_F = np.clip(current_F, F_min, F_max)
                
                current_CR = mean_CR + 0.1 * np.random.normal(0, 1)
                current_CR = np.clip(current_CR, CR_min, CR_max)
                
                # Strategy adaptation - alternate between strategies
                strategy = np.random.choice(['rand/1', 'best/1', 'current-to-best/1', 'rand/2'], 
                                            p=[0.25, 0.25, 0.25, 0.25])
                
                if strategy == 'rand/1':
                    # DE/rand/1
                    a, b, c = random.sample([x for j, x in enumerate(population) if j != i], 3)
                    mutant = np.clip(a.vector + current_F * (b.vector - c.vector), *BOUNDS)
                    parents = [a, b, c]
                    
                elif strategy == 'best/1':
                    # DE/best/1
                    best = sorted_pop[0]
                    a, b = random.sample([x for j, x in enumerate(population) if j != i], 2)
                    mutant = np.clip(best.vector + current_F * (a.vector - b.vector), *BOUNDS)
                    parents = [best, a, b]
                    
                elif strategy == 'current-to-best/1':
                    # DE/current-to-best/1
                    best = sorted_pop[0]
                    a, b = random.sample([x for j, x in enumerate(population) if j != i], 2)
                    mutant = np.clip(population[i].vector + current_F * (best.vector - population[i].vector) + 
                                    current_F * (a.vector - b.vector), *BOUNDS)
                    parents = [population[i], best, a, b]
                    
                else:  # rand/2
                    # DE/rand/2
                    a, b, c, d, e = random.sample([x for j, x in enumerate(population) if j != i], 5)
                    mutant = np.clip(a.vector + current_F * (b.vector - c.vector) + 
                                    current_F * (d.vector - e.vector), *BOUNDS)
                    parents = [a, b, c, d, e]
                
                # Crossover - use binomial crossover with occasional exponential
                if np.random.random() < 0.9:  # 90% binomial
                    trial = np.array([mutant[j] if random.random() < current_CR or j == np.random.randint(DIM) 
                                    else population[i].vector[j] for j in range(DIM)])
                else:  # 10% exponential
                    trial = population[i].vector.copy()
                    j = np.random.randint(DIM)
                    L = 0
                    while (random.random() < current_CR and L < DIM):
                        trial[j] = mutant[j]
                        j = (j + 1) % DIM
                        L += 1
                
                trial_fitness = cec2021_f(trial)
                
                # Log operator attribution
                if len(parents) >= 3:  # Basic operators always have at least 3 parents
                    OAM["DE"]["mutation"][-1].append(max(p.fitness for p in parents) - cec2021_f(mutant))
                OAM["DE"]["crossover"][-1].append(cec2021_f(mutant) - trial_fitness)
                
                child = EvoIndividual(trial, trial_fitness, "DE", [x.id for x in parents], f"DE/{strategy}")
                log_to_peg(child, parents)
                
                # Selection with memory
                if child.fitness < population[i].fitness:
                    new_pop.append(child)
                    # Record successful parameters
                    successful_F.append(current_F)
                    successful_CR.append(current_CR)
                else:
                    new_pop.append(population[i])
            
            population = new_pop
            
            # Update parameter adaptation memory
            if successful_F:
                success_history["F"].extend(successful_F)
                success_history["CR"].extend(successful_CR)
                # Keep only recent history
                success_history["F"] = success_history["F"][-50:]
                success_history["CR"] = success_history["CR"][-50:]
            
            best = min(population, key=lambda x: x.fitness)
            fitness_curve.append(best.fitness)
            
            if best.fitness < best_fitness - 1e-6:
                best_fitness = best.fitness
                no_improve = 0
            else:
                no_improve += 1
                
                # Add restart mechanism without early termination
                if no_improve >= ESC_TOLERANCE // 2:
                    # Keep best individual
                    elite = population[0]
                    # Reinitialize 40% of population
                    for i in range(1, int(0.4 * POP_SIZE)):
                        population[i].vector = np.random.uniform(*BOUNDS, DIM)
                        population[i].fitness = cec2021_f(population[i].vector)
                    no_improve = 0
    
        return fitness_curve
    
    def calculate_cds(OAM):
        """
        Calculate Convergence Driver Score (CDS) for all operators.
        
        Args:
            OAM: Operator Attribution Matrix data
            
        Returns:
            Three nested dictionaries: 
            - per_iteration_cds: CDS values for each operator at each iteration
            - total_cds: Raw CDS values across all iterations
            - operator_cds: Raw CDS values for each operator
        """
        per_iteration_cds = {}
        total_cds = {}
        operator_cds = {}
        
        # Calculate CDS for each iteration
        for algo, operators in OAM.items():
            per_iteration_cds[algo] = {}
            total_cds[algo] = {}
            
            for op_name, iterations in operators.items():
                per_iteration_cds[algo][op_name] = []
                
                # Calculate CDS for each iteration
                for iter_data in iterations:
                    if iter_data and len(iter_data) > 0:
                        # CDS(t)Oₖ = (1/n(t)Oₖ) × Σ ∆f(t)Oₖ,j
                        cds_value = sum(iter_data) / len(iter_data)
                    else:
                        cds_value = 0
                        
                    per_iteration_cds[algo][op_name].append(cds_value)
                
                # Total CDS across all iterations
                total_cds[algo][op_name] = sum(per_iteration_cds[algo][op_name])
        
        # Store raw CDS values without normalization
        for algo, operators in total_cds.items():
            operator_cds[algo] = {}
            for op_name, value in operators.items():
                operator_cds[algo][op_name] = value
        
        return per_iteration_cds, total_cds, operator_cds
    def plot_cds_bar_charts(operator_cds):
        """
        Create bar charts visualizing the raw CDS values for each algorithm.
        
        Args:
            operator_cds: Dictionary of raw CDS values
        """
        # Define colors matching the line plots
        colors = {
            "GA": {
                "crossover": "blue",
                "mutation": "deepskyblue"
            },
            "PSO": {
                "cognitive_component": "limegreen",
                "social_component": "green",
                "inertia_component": "darkgreen",
                "position_update": "springgreen"
            },
            "CS": {
                "levy_flight": "orange",
                "successful_replacement": "darkorange",
                "unsuccessful_attempt": "gold",
                "abandoned_nest": "saddlebrown",
                "random_init": "peru"
            },
            "DE": {
                "mutation": "red",
                "crossover": "darkred"
            }
        }
        
        # Combined bar chart for all algorithms
        plt.figure(figsize=(20, 10))
        
        # Collect all algorithms and their operators
        all_algos = list(operator_cds.keys())
        all_values = []
        for algo in all_algos:
            all_values.extend(operator_cds[algo].values())
        
        positions = np.arange(len(all_algos))
        width = 1.0 / max(len(ops) for ops in operator_cds.values())
        
        # Plot bars for each algorithm and operator
        for algo_idx, algo in enumerate(all_algos):
            operators = operator_cds[algo]
            sorted_operators = sorted(operators.items(), key=lambda x: x[1], reverse=True)
            
            for op_idx, (op_name, value) in enumerate(sorted_operators):
                pos = positions[algo_idx] + (op_idx - len(operators)/2 + 0.5) * width
                bar = plt.bar(pos, value, width=width, 
                        color=colors[algo].get(op_name, 'gray'),
                        edgecolor='black', linewidth=1, alpha=0.8,
                        label=f"{algo}-{op_name}")
                
                # Add value label above or below bars depending on value
                if abs(value) > 1e-10:  # Show all non-zero values
                    label_pos = value + np.sign(value) * max(abs(max(all_values)), abs(min(all_values))) * 0.01
                    va = 'bottom' if value >= 0 else 'top'
                    plt.text(pos, label_pos, f'{value:.2f}', 
                            ha='center', va=va, fontsize=10, rotation=45)
    
        plt.title('Raw Convergence Driver Score (CDS) Across Algorithms', fontsize=16)
        plt.ylabel('CDS Value', fontsize=14)
        plt.xticks(positions, all_algos, fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis limits with proper padding for both positive and negative values
        max_val = max(all_values) if all_values else 1.0
        min_val = min(all_values) if all_values else 0.0
        y_padding = max(abs(max_val), abs(min_val)) * 0.2  # 20% padding
        plt.ylim(min_val - y_padding, max_val + y_padding)
        
        # Use consistent layout approach
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, ncol=1)
        plt.tight_layout()
        # Adjust the layout to accommodate the legend
        plt.gcf().set_size_inches(12, 8)
        plt.show()
        
        # Individual bar charts for each algorithm
        for algo, operators in operator_cds.items():
            if not operators:
                continue
                
            plt.figure(figsize=(12, 8))  # Increased figure size
            
            # Sort operators by CDS value
            sorted_operators = sorted(operators.items(), key=lambda x: x[1], reverse=True)
            op_names = [op[0] for op in sorted_operators]
            values = [op[1] for op in sorted_operators]
            
            # Create bars with algorithm-specific colors
            bars = plt.bar(op_names, values, 
                          color=[colors[algo].get(op, 'gray') for op in op_names],
                          edgecolor='black', linewidth=1)
            
            # Add value labels with adjusted position for positive and negative values
            max_val = max(values) if values else 1.0
            min_val = min(values) if values else 0.0
            label_offset = max(abs(max_val), abs(min_val)) * 0.02  # 2% of max absolute value
            
            for bar, val in zip(bars, values):
                if abs(val) > 1e-10:  # Show non-zero values
                    label_pos = val + np.sign(val) * label_offset
                    va = 'bottom' if val >= 0 else 'top'
                    plt.text(bar.get_x() + bar.get_width()/2, label_pos, 
                            f'{val:.3f}', ha='center', va=va, fontsize=10)
            
            plt.title(f'{algo} Raw Convergence Driver Score (CDS)', fontsize=14)
            plt.ylabel('CDS Value', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set y-axis limits with proper padding for both positive and negative values
            min_val = min(values) if values else 0.0
            if np.isclose(max_val, 0, atol=1e-10) and np.isclose(min_val, 0, atol=1e-10):
                plt.ylim(-1.0, 1.0)  # Default range if all values are 0
            else:
                y_padding = max(abs(max_val), abs(min_val)) * 0.2  # 20% padding
                plt.ylim(min_val - y_padding, max_val + y_padding)
            
            # Use consistent layout approach
            plt.tight_layout()
            plt.show()               
    def plot_cds_time_series(per_iteration_cds):
        """
        Plot CDS values over iterations to show dynamics of operator contribution.
        
        Args:
            per_iteration_cds: Dictionary of CDS values per iteration
        """
        # Define colors matching the line plots
        colors = {
            "GA": {
                "crossover": "blue",
                "mutation": "deepskyblue"
            },
            "PSO": {
                "cognitive_component": "limegreen",
                "social_component": "green",
                "inertia_component": "darkgreen",
                "position_update": "springgreen"
            },
            "CS": {
                "levy_flight": "orange",
                "successful_replacement": "darkorange",
                "unsuccessful_attempt": "gold",
                "abandoned_nest": "saddlebrown",
                "random_init": "peru"
            },
            "DE": {
                "mutation": "red",
                "crossover": "darkred"
            }
        }
        
        # Create a time series plot for each algorithm
        for algo, operators in per_iteration_cds.items():
            plt.figure(figsize=(12, 6))
            
            for op_name, values in operators.items():
                # Add some basic smoothing to help with readability
                # Use moving average if we have enough points
                if len(values) > 5:
                    smoothed = np.convolve(values, np.ones(5)/5, mode='valid')
                    plt.plot(np.arange(len(smoothed)), smoothed, 
                            label=op_name, color=colors[algo].get(op_name, 'gray'), 
                            linewidth=2, alpha=0.8)
                else:
                    plt.plot(np.arange(len(values)), values, 
                            label=op_name, color=colors[algo].get(op_name, 'gray'), 
                            linewidth=2, alpha=0.8)
            
            plt.title(f'{algo} Convergence Driver Score (CDS) Over Iterations', fontsize=14)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('CDS Value', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
            plt.tight_layout()
            plt.show()
    
    
    # ==== Run All Algorithms ====
    fitness_ga = ga_run()
    fitness_pso = pso_run()
    fitness_cs = cs_run()
    fitness_de = de_run()
    
    
    # ==== Print Best Final Fitness for Each Algorithm ====
    print("\nBest Final Fitness Values:")
    print(f"GA: {fitness_ga[-1]:.4f}")
    print(f"PSO: {fitness_pso[-1]:.4f}")
    print(f"CS: {fitness_cs[-1]:.4f}")
    print(f"DE: {fitness_de[-1]:.4f}")
    
    
    
    # ==== Fitness Convergence Plot ====
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_ga, label="GA")
    plt.plot(fitness_pso, label="PSO")
    plt.plot(fitness_cs, label="CS")
    plt.plot(fitness_de, label="DE")
    plt.title(f"CEC2021 {selected_func_name} Function Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()
    plt.show()
    
    # ==== Calculate and Plot CDS ====
    per_iteration_cds, total_cds, normalized_cds = calculate_cds(OAM)
    plot_cds_bar_charts(normalized_cds)
    plot_cds_time_series(per_iteration_cds)
    # ==== OAM Plotting Functions ====
    def pad_with_nan(lst, length):
        return [np.mean(x) if x else np.nan for x in lst] + [np.nan] * (length - len(lst))
    
    def plot_individual_oams(OAM):
        colors = {
            "GA": {
                "crossover": "blue",
                "mutation": "deepskyblue"
            },
            "PSO": {
                "cognitive_component": "limegreen",
                "social_component": "green",
                "inertia_component": "darkgreen",
                "position_update": "springgreen"
            },
            "CS": {
                "levy_flight": "orange",
                "successful_replacement": "darkorange",
                "unsuccessful_attempt": "gold",
                "abandoned_nest": "saddlebrown",
                "random_init": "peru"
            },
            "DE": {
                "mutation": "red",
                "crossover": "darkred"
            }
        }
        max_iters = max(len(vals) for algo_ops in OAM.values() for vals in algo_ops.values())
        
        for algo, ops in OAM.items():
            plt.figure(figsize=(10, 6))
            for op, values in ops.items():
                padded = pad_with_nan(values, max_iters)
                plt.plot(padded, label=op, color=colors[algo].get(op, "black"))
            plt.title(f"{algo} Operator Attribution")
            plt.xlabel("Iteration")
            plt.ylabel("Avg Fitness Gain")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    # Plot individual OAMs
    plot_individual_oams(OAM)
    # ==== OAM Bar Chart Visualization ====
    # ==== Individual OAM Bar Chart Visualizations ====
    def plot_individual_oam_bar_charts(OAM):
    # Calculate average contribution per operator for each algorithm
        algorithm_operator_means = {}
        
        for algo, ops in OAM.items():
            algorithm_operator_means[algo] = {}
            for op, values in ops.items():
                # Calculate the mean of all non-empty sublists' means 
                non_empty_means = [np.mean(sublist) for sublist in values if sublist]
                if non_empty_means:
                    algorithm_operator_means[algo][op] = np.mean(non_empty_means)
                else:
                    algorithm_operator_means[algo][op] = 0
        
        # Define colors matching the line plots
        colors = {
            "GA": {
                "crossover": "blue",
                "mutation": "deepskyblue"
            },
            "PSO": {
                "cognitive_component": "limegreen",
                "social_component": "green",
                "inertia_component": "darkgreen",
                "position_update": "springgreen"
            },
            "CS": {
                "levy_flight": "orange",
                "successful_replacement": "darkorange",
                "unsuccessful_attempt": "gold",
                "abandoned_nest": "saddlebrown",
                "random_init": "peru"
            },
            "DE": {
                "mutation": "red",
                "crossover": "darkred"
            }
        }
        
        # Create a separate figure for each algorithm
        for algo, op_means in algorithm_operator_means.items():
            # Skip if all values are effectively zero
            if all(abs(v) < 1e-10 for v in op_means.values()):
                print(f"Skipping {algo} plot - all values are effectively zero")
                continue
                
            # Create new figure with adjusted size
            plt.figure(figsize=(6, 4))
            
            # Extract operators and their mean values
            operators = list(op_means.keys())
            means = list(op_means.values())
            
            # Create bar colors list
            bar_colors = [colors[algo].get(op, 'gray') for op in operators]
            
            # Create bars
            plt.bar(operators, means, color=bar_colors, width=0.6)
            
            # Customize plot
            plt.title(f"{algo} Operator Average Contribution", pad=15)
            plt.ylabel("Average Fitness Improvement")
            
            # Adjust x-axis labels
            plt.xticks(rotation=30, ha='right')
            
            # Add grid
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set y-axis limits with proper padding for both positive and negative values
            y_vals = [v for v in means if abs(v) > 1e-10]  # Filter out near-zero values
            if y_vals:
                y_min, y_max = min(y_vals), max(y_vals)
                if y_min >= 0:
                    plt.ylim(0, y_max * 1.15)
                else:
                    # Add padding to both positive and negative sides
                    padding = max(abs(y_min), abs(y_max)) * 0.15
                    plt.ylim(y_min - padding, y_max + padding)
            else:
                plt.ylim(-1, 1)  # Default range if all values are near zero
            
            # Use consistent layout approach
            plt.tight_layout()
            plt.show()
               
    # Add this call after the other plotting functions
    plot_individual_oam_bar_charts(OAM)
    # ==== PEG Visualization ====
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(PEG, seed=1)
    
    algo_colors = {
        "GA": "blue",
        "PSO": "green",
        "CS": "orange",
        "DE": "red"
    }
    node_colors = []
    for n in PEG.nodes:
        algo = PEG.nodes[n].get('algo', 'UNKNOWN')
        node_colors.append(algo_colors.get(algo, 'gray'))
    
    nx.draw(
        PEG,
        pos,
        node_color=node_colors,
        node_size=30,
        edge_color='gray',
        alpha=0.6
    )
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=a) for a, c in algo_colors.items()
    ] + [Patch(facecolor='gray', label='UNKNOWN')]
    
    plt.legend(handles=legend_elements, loc='upper right', title="Algorithm")
    plt.title("Population Evolution Graph (PEG) - CEC2021 F1")
    plt.axis('off')
    plt.show()
    
    for algo_name, algo_peg in PEGs.items():
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(algo_peg, seed=1)
    
        color = algo_colors.get(algo_name, 'gray')
        node_colors = [color for _ in algo_peg.nodes]
    
        nx.draw(
            algo_peg,
            pos,
            node_color=node_colors,
            node_size=30,
            edge_color='gray',
            alpha=0.6
        )
    
        plt.title(f"Population Evolution Graph (PEG) - {algo_name}")
        plt.axis('off')
        plt.show()
    
    
    
    def generate_comparative_explanations(OAM, fitness_curves, selected_func_name, PEG, PEGs):
        """
        Generate detailed textual explanations comparing algorithm performance and operator contributions
        based on EvoMapX framework data.
        
        Args:
            OAM: Operator Attribution Matrix data
            fitness_curves: Dictionary of fitness curves for each algorithm
            selected_func_name: Name of the CEC function being optimized
            PEG: Combined Population Evolution Graph
            PEGs: Individual Population Evolution Graphs for each algorithm
        """
        # ===== Performance Comparison =====
        performance_explanation = explain_performance_comparison(fitness_curves, selected_func_name)
        
        # ===== Operator Contribution Analysis =====
        operator_explanation = explain_operator_contributions(OAM)
        
        # ===== Convergence Pattern Analysis =====
        convergence_explanation = explain_convergence_patterns(fitness_curves, OAM)
        
        # ===== Algorithm-Specific Analysis =====
        algorithm_explanations = explain_algorithm_specific_behaviors(OAM, fitness_curves, PEGs)
        
        # ===== PEG Structure Analysis =====
        peg_explanation = explain_peg_structures(PEG, PEGs)
        
        # ===== Print All Explanations =====
        print("\n" + "="*80)
        print("                    EvoMapX COMPARATIVE TEXTUAL ANALYSIS")
        print("="*80 + "\n")
        
        print("FUNCTION: ", selected_func_name)
        print("\n" + "-"*80)
        print("1. ALGORITHM PERFORMANCE COMPARISON")
        print("-"*80)
        print(performance_explanation)
        
        print("\n" + "-"*80)
        print("2. OPERATOR CONTRIBUTION ANALYSIS")
        print("-"*80)
        print(operator_explanation)
        
        print("\n" + "-"*80)
        print("3. CONVERGENCE PATTERN ANALYSIS")
        print("-"*80)
        print(convergence_explanation)
        
        print("\n" + "-"*80)
        print("4. ALGORITHM-SPECIFIC BEHAVIOR ANALYSIS")
        print("-"*80)
        for algo, explanation in algorithm_explanations.items():
            print(f"\n-- {algo} ANALYSIS --")
            print(explanation)
        
        print("\n" + "-"*80)
        print("5. POPULATION EVOLUTION STRUCTURE ANALYSIS")
        print("-"*80)
        print(peg_explanation)
        
        # Extend comparative explanations with CDS insights
        print("\n" + "-"*80)
        print("6. CONVERGENCE DRIVER SCORE ANALYSIS:")
        print("-"*80)
        for algo, curve in fitness_curves.items():
            if algo in normalized_cds:
                top_driver = max(normalized_cds[algo].items(), key=lambda x: x[1])
                print(f"- {algo} achieved final fitness {curve[-1]:.4e} with {top_driver[0]} as its main convergence driver ({top_driver[1]:.2f})")
        
        print("\n" + "="*80)
        print("                       END OF EVOMAPX ANALYSIS")
        print("="*80)
    
    def explain_performance_comparison(fitness_curves, func_name):
        """Generate explanation comparing performance of all algorithms."""
        
        # Final best fitness values
        final_fitness = {algo: curve[-1] for algo, curve in fitness_curves.items()}
        best_algo = min(final_fitness, key=final_fitness.get)
        worst_algo = max(final_fitness, key=final_fitness.get)
        
        # Convergence speed (iterations to reach 90% of final improvement)
        convergence_speed = {}
        for algo, curve in fitness_curves.items():
            initial = curve[0]
            final = curve[-1]
            target = final + 0.1 * (initial - final)  # 90% of improvement
            
            for i, fitness in enumerate(curve):
                if fitness <= target:
                    convergence_speed[algo] = i
                    break
            else:
                convergence_speed[algo] = len(curve)
        
        fastest_algo = min(convergence_speed, key=convergence_speed.get)
        slowest_algo = max(convergence_speed, key=convergence_speed.get)
        
        # Format results for ranking
        algo_ranking = sorted(final_fitness.keys(), key=lambda x: final_fitness[x])
        
        explanation = f"For the {func_name} function:\n\n"
        explanation += f"- {best_algo} achieved the best fitness value of {final_fitness[best_algo]:.4e}, "
        explanation += f"while {worst_algo} had the worst fitness of {final_fitness[worst_algo]:.4e}.\n\n"
        
        explanation += "Performance ranking from best to worst:\n"
        for i, algo in enumerate(algo_ranking):
            explanation += f"  {i+1}. {algo}: {final_fitness[algo]:.4e}\n"
        
        explanation += f"\n{fastest_algo} demonstrated the fastest convergence, reaching 90% of its final improvement "
        explanation += f"in {convergence_speed[fastest_algo]} iterations, compared to {convergence_speed[slowest_algo]} "
        explanation += f"for the slowest converging algorithm ({slowest_algo}).\n\n"
        
        # Early vs late phase comparison
        early_improvement = {}
        late_improvement = {}
        
        for algo, curve in fitness_curves.items():
            mid_point = len(curve) // 2
            early_imp = (curve[0] - curve[mid_point]) / curve[0] if curve[0] != 0 else 0
            late_imp = (curve[mid_point] - curve[-1]) / curve[mid_point] if curve[mid_point] != 0 else 0
            
            early_improvement[algo] = early_imp
            late_improvement[algo] = late_imp
        
        best_early = max(early_improvement, key=early_improvement.get)
        best_late = max(late_improvement, key=late_improvement.get)
        
        explanation += f"{best_early} showed the strongest initial descent ({early_improvement[best_early]:.2%} improvement), "
        explanation += f"while {best_late} demonstrated the best refinement in later iterations "
        explanation += f"({late_improvement[best_late]:.2%} improvement in the second half)."
        
        return explanation
    
    def calculate_operator_contributions(OAM):
        """Calculate average contribution of each operator across iterations."""
        contributions = {}
        
        for algo, operators in OAM.items():
            contributions[algo] = {}
            for op, values in operators.items():
                # Calculate mean of non-empty lists
                non_empty_values = [np.mean(v) for v in values if v]
                if non_empty_values:
                    contributions[algo][op] = np.mean(non_empty_values)
                else:
                    contributions[algo][op] = 0
        
        return contributions
    
    def explain_operator_contributions(OAM):
        """Generate explanation of operator contributions based on OAM."""
        contributions = calculate_operator_contributions(OAM)
        
        # Find most important operators for each algorithm
        top_operators = {}
        for algo, ops in contributions.items():
            sorted_ops = sorted(ops.items(), key=lambda x: x[1], reverse=True)
            top_operators[algo] = sorted_ops[:2]  # Top 2 operators
        
        # Calculate aggregate contribution across algorithms
        all_operators = []
        for algo, ops in contributions.items():
            all_operators.extend([(algo, op, val) for op, val in ops.items()])
        
        # Sort by contribution value
        all_operators.sort(key=lambda x: x[2], reverse=True)
        
        explanation = "Analysis of operator contributions across all algorithms:\n\n"
        
        # Individual algorithm analysis
        for algo, ops in top_operators.items():
            if not ops:
                continue
            explanation += f"In {algo}, the most effective operators were:\n"
            for op, val in ops:
                explanation += f"- {op}: average fitness gain of {val:.4e}\n"
            explanation += "\n"
        
        # Cross-algorithm comparison
        if all_operators:
            explanation += "Comparing operators across algorithms:\n"
            for algo, op, val in all_operators[:5]:  # Top 5 overall
                explanation += f"- {algo}'s {op}: {val:.4e}\n"
            
            explanation += "\nThe overall most effective search mechanism was "
            explanation += f"{all_operators[0][0]}'s {all_operators[0][1]} "
            explanation += f"with average fitness gain of {all_operators[0][2]:.4e}.\n"
        
        return explanation
    
    def explain_convergence_patterns(fitness_curves, OAM):
        """Generate explanation of convergence patterns."""
        explanation = "Convergence pattern analysis:\n\n"
        
        # Analyze convergence phases
        for algo, curve in fitness_curves.items():
            # Calculate improvement rates
            improvements = []
            for i in range(1, len(curve)):
                improvements.append((curve[i-1] - curve[i]) / max(1e-10, curve[i-1]))
            
            # Identify stagnation phases
            stagnation_threshold = 1e-4
            stagnation_phases = []
            current_stagnation = None
            
            for i, imp in enumerate(improvements):
                if imp < stagnation_threshold:
                    if current_stagnation is None:
                        current_stagnation = [i]
                else:
                    if current_stagnation is not None:
                        current_stagnation.append(i)
                        stagnation_phases.append(current_stagnation)
                        current_stagnation = None
            
            # Find phases with rapid improvement
            rapid_threshold = np.percentile(improvements, 90)  # Top 10% improvements
            rapid_phases = []
            
            for i, imp in enumerate(improvements):
                if imp >= rapid_threshold:
                    rapid_phases.append(i)
            
            explanation += f"{algo} convergence pattern:\n"
            explanation += f"- Initial fitness: {curve[0]:.4e}\n"
            explanation += f"- Final fitness: {curve[-1]:.4e}\n"
            explanation += f"- Total improvement: {(curve[0] - curve[-1])/curve[0]:.2%}\n"
            
            if rapid_phases:
                explanation += f"- Rapid improvement occurred during iterations: {', '.join(map(str, rapid_phases[:5]))}"
                if len(rapid_phases) > 5:
                    explanation += f" and {len(rapid_phases)-5} more"
                explanation += "\n"
            
            if stagnation_phases:
                explanation += f"- {len(stagnation_phases)} stagnation phases identified\n"
                if len(stagnation_phases) <= 3:
                    for i, phase in enumerate(stagnation_phases):
                        explanation += f"  * Phase {i+1}: iterations {phase[0]} to {phase[1]}\n"
            
            explanation += "\n"
        
        return explanation
    
    def explain_algorithm_specific_behaviors(OAM, fitness_curves, PEGs):
        """Generate algorithm-specific behavior explanations."""
        explanations = {}
        
        # Calculate operator contributions over time
        op_contributions = {}
        for algo, operators in OAM.items():
            op_contributions[algo] = defaultdict(list)
            for op, values in operators.items():
                for i, val_list in enumerate(values):
                    if val_list:  # Make sure the list is not empty
                        op_contributions[algo][op].append((i, np.mean(val_list)))
        
        # GA-specific analysis
        if "GA" in OAM:
            explanations["GA"] = "Genetic Algorithm exhibited "
            
            # Compare crossover vs mutation effectiveness
            ga_contributions = calculate_operator_contributions({"GA": OAM["GA"]})["GA"]
            if ga_contributions.get("crossover", 0) > ga_contributions.get("mutation", 0):
                explanations["GA"] += "stronger fitness improvement from crossover than mutation, "
                explanations["GA"] += "suggesting that recombination of existing genetic material was more effective "
                explanations["GA"] += "than random modifications for this problem.\n\n"
            else:
                explanations["GA"] += "stronger fitness improvement from mutation than crossover, "
                explanations["GA"] += "suggesting that exploration of new genetic material was more effective "
                explanations["GA"] += "than recombination for this problem.\n\n"
            
            # Analyze convergence behavior
            curve = fitness_curves.get("GA", [])
            if curve:
                mid_point = len(curve) // 2
                early_imp = curve[0] - curve[mid_point]
                late_imp = curve[mid_point] - curve[-1]
                
                if early_imp > late_imp * 3:
                    explanations["GA"] += "The GA showed rapid early convergence followed by diminishing returns, "
                    explanations["GA"] += "which is typical for problems with many local optima where the population "
                    explanations["GA"] += "quickly finds good regions but struggles to refine solutions further."
                else:
                    explanations["GA"] += "The GA maintained steady improvement throughout the search process, "
                    explanations["GA"] += "indicating good balance between exploration and exploitation phases."
        
        # PSO-specific analysis
        if "PSO" in OAM:
            explanations["PSO"] = "Particle Swarm Optimization demonstrated "
            
            # Compare cognitive, social and inertia components
            pso_contributions = calculate_operator_contributions({"PSO": OAM["PSO"]})["PSO"]
            components = {
                "cognitive_component": pso_contributions.get("cognitive_component", 0),
                "social_component": pso_contributions.get("social_component", 0),
                "inertia_component": pso_contributions.get("inertia_component", 0)
            }
            
            dominant_component = max(components, key=components.get)
            if dominant_component == "cognitive_component":
                explanations["PSO"] += "stronger reliance on individual particle memories (cognitive component), "
                explanations["PSO"] += "suggesting particles benefited more from their personal experiences "
                explanations["PSO"] += "than from swarm knowledge.\n\n"
            elif dominant_component == "social_component":
                explanations["PSO"] += "stronger reliance on swarm intelligence (social component), "
                explanations["PSO"] += "indicating effective information sharing among particles and "
                explanations["PSO"] += "convergence toward promising regions.\n\n"
            else:
                explanations["PSO"] += "significant momentum-driven search (inertia component), "
                explanations["PSO"] += "suggesting particles maintained their trajectories to effectively "
                explanations["PSO"] += "explore the search space before converging.\n\n"
            
            # Velocity analysis from graph structure
            if "PSO" in PEGs:
                pso_peg = PEGs["PSO"]
                avg_deg = np.mean([d for _, d in pso_peg.degree()])
                explanations["PSO"] += f"The average connectivity in the PSO evolution graph was {avg_deg:.2f}, "
                if avg_deg < 2:
                    explanations["PSO"] += "indicating linear particle trajectories with few branching behaviors. "
                    explanations["PSO"] += "This suggests a straightforward convergence pattern."
                else:
                    explanations["PSO"] += "showing complex interconnected particle behaviors. "
                    explanations["PSO"] += "This suggests rich exploratory dynamics."
        
        # CS-specific analysis
        if "CS" in OAM:
            explanations["CS"] = "Cuckoo Search algorithm showed "
            
            # Compare Lévy flights vs abandonment effectiveness
            cs_contributions = calculate_operator_contributions({"CS": OAM["CS"]})["CS"]
            levy_effect = cs_contributions.get("levy_flight", 0)
            abandon_effect = cs_contributions.get("abandoned_nest", 0) + cs_contributions.get("random_init", 0)
            
            if levy_effect > abandon_effect:
                explanations["CS"] += "stronger improvement from Lévy flight steps than nest abandonment, "
                explanations["CS"] += "indicating that local search was more effective than global resets.\n\n"
            else:
                explanations["CS"] += "stronger improvement from nest abandonment than Lévy flights, "
                explanations["CS"] += "suggesting the algorithm benefited from periodic restarts to escape local optima.\n\n"
            
            # Success rate analysis
            if "successful_replacement" in OAM["CS"] and "unsuccessful_attempt" in OAM["CS"]:
                successful = sum(1 for vals in OAM["CS"]["successful_replacement"] if vals)
                unsuccessful = sum(1 for vals in OAM["CS"]["unsuccessful_attempt"] if vals)
                total = successful + unsuccessful
                if total > 0:
                    success_rate = successful / total
                    explanations["CS"] += f"The success rate for new solution acceptance was {success_rate:.2%}, "
                    if success_rate > 0.3:
                        explanations["CS"] += "which is relatively high and indicates an effective search strategy "
                        explanations["CS"] += "for this problem landscape."
                    else:
                        explanations["CS"] += "which is relatively low and suggests the algorithm struggled to find "
                        explanations["CS"] += "consistently better solutions, possibly due to a rugged fitness landscape."
        
        # DE-specific analysis
        if "DE" in OAM:
            explanations["DE"] = "Differential Evolution exhibited "
            
            # Compare mutation vs crossover effectiveness
            de_contributions = calculate_operator_contributions({"DE": OAM["DE"]})["DE"]
            if de_contributions.get("mutation", 0) > de_contributions.get("crossover", 0):
                explanations["DE"] += "stronger fitness improvement from mutation than crossover, "
                explanations["DE"] += "indicating that the differential vector creation was particularly effective "
                explanations["DE"] += "for navigating this function's landscape.\n\n"
            else:
                explanations["DE"] += "stronger fitness improvement from crossover than mutation, "
                explanations["DE"] += "suggesting that selective combination of mutant and target vectors was key "
                explanations["DE"] += "to finding better solutions.\n\n"
            
            # Analyze convergence stability
            curve = fitness_curves.get("DE", [])
            if curve and len(curve) > 5:
                diffs = [abs(curve[i] - curve[i-1]) for i in range(1, len(curve))]
                stability = np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else 0
                
                explanations["DE"] += f"The convergence stability coefficient was {stability:.4f}, "
                if stability < 1.0:
                    explanations["DE"] += "indicating smooth and steady progress toward optimal solutions. "
                    explanations["DE"] += "This suggests the algorithm maintained a good balance between "
                    explanations["DE"] += "exploration and exploitation throughout the search process."
                else:
                    explanations["DE"] += "showing variability in convergence rates across generations. "
                    explanations["DE"] += "This suggests periods of rapid improvement interspersed with "
                    explanations["DE"] += "phases of slower progress, typical for complex fitness landscapes."
                    
        return explanations
    
    def explain_peg_structures(PEG, PEGs):
        """Generate explanation of population evolution graph structures."""
        explanation = "Population Evolution Graph (PEG) structural analysis:\n\n"
        
        # Analyze overall PEG
        nodes = PEG.number_of_nodes()
        edges = PEG.number_of_edges()
        density = edges / (nodes * (nodes - 1)) if nodes > 1 else 0
        
        explanation += f"The combined PEG contains {nodes} unique individuals connected by {edges} evolutionary relationships, "
        explanation += f"with a graph density of {density:.6f}.\n\n"
        
        # Analyze individual algorithm PEGs
        for algo, graph in PEGs.items():
            if graph.number_of_nodes() == 0:
                continue
                
            algo_nodes = graph.number_of_nodes()
            algo_edges = graph.number_of_edges()
            
            # Calculate average in/out degree
            in_degrees = [d for _, d in graph.in_degree()]
            out_degrees = [d for _, d in graph.out_degree()]
            avg_in = np.mean(in_degrees) if in_degrees else 0
            avg_out = np.mean(out_degrees) if out_degrees else 0
            
            # Identify nodes with highest in-degree (most influential)
            if in_degrees:
                max_in_degree = max(in_degrees) if in_degrees else 0
                influential_count = sum(1 for d in in_degrees if d == max_in_degree)
            else:
                max_in_degree = 0
                influential_count = 0
            
            explanation += f"{algo} evolution structure:\n"
            explanation += f"- Contains {algo_nodes} individuals with {algo_edges} parent-child relationships\n"
            explanation += f"- Average parent count per individual: {avg_in:.2f}\n"
            explanation += f"- Average child count per individual: {avg_out:.2f}\n"
            
            if influential_count > 0:
                explanation += f"- {influential_count} individuals had the highest influence (in-degree {max_in_degree})\n"
            
            # Interpret the structure
            if avg_in > 1.5:
                explanation += "- Shows complex ancestral patterns with multiple parents contributing to offspring\n"
            else:
                explanation += "- Displays mostly linear evolution with limited genetic mixing\n"
                
            # Assess structural diversity
            if density > 0.05:
                explanation += "- High connectivity suggests strong information sharing across the population\n"
            else:
                explanation += "- Sparse connectivity indicates largely independent evolutionary paths\n"
            
            explanation += "\n"
        
        return explanation
    
    #  CDS metrics and their visualizion
    
    
    
    
    
    def explain_cds_analysis(total_cds, normalized_cds):
        """
        Generate textual explanation of CDS analysis results.
        
        Args:
            total_cds: Dictionary of total CDS values
            normalized_cds: Dictionary of normalized CDS values
            
        Returns:
            String containing detailed CDS analysis
        """
        explanation = "\n" + "-"*80 + "\n"
        explanation += "CONVERGENCE DRIVER SCORE (CDS) ANALYSIS\n"
        explanation += "-"*80 + "\n\n"
        
        # Overall top contributors
        all_operators = []
        for algo, operators in normalized_cds.items():
            for op_name, value in operators.items():
                all_operators.append((algo, op_name, value))
        
        all_operators.sort(key=lambda x: x[2], reverse=True)
        
        explanation += "Top convergence drivers across all algorithms:\n"
        for i, (algo, op, val) in enumerate(all_operators[:5]):
            explanation += f"{i+1}. {algo}'s {op}: {val:.3f} normalized CDS\n"
        
        explanation += "\nConvergence driver analysis by algorithm:\n"
        
        # Per-algorithm analysis
        for algo, operators in normalized_cds.items():
            # Skip if empty
            if not operators:
                continue
                
            sorted_ops = sorted(operators.items(), key=lambda x: x[1], reverse=True)
            
            explanation += f"\n-- {algo} ANALYSIS --\n"
            explanation += f"Primary convergence driver: {sorted_ops[0][0]} ({sorted_ops[0][1]:.3f})\n"
            
            # Show distribution of driving forces
            explanation += "CDS distribution:\n"
            for op, val in sorted_ops:
                explanation += f"- {op}: {val:.3f} ({val*100:.1f}%)\n"
            
            # Interpretation
            if len(sorted_ops) > 1:
                # Calculate dominance ratio of top operator vs second
                ratio = sorted_ops[0][1] / max(sorted_ops[1][1], 0.001)
                
                if ratio > 2:
                    explanation += f"\nThe {sorted_ops[0][0]} operator dominates convergence "
                    explanation += f"({ratio:.1f}x stronger than the next operator), "
                    explanation += "suggesting optimization is heavily reliant on a single search mechanism.\n"
                elif ratio > 1.3:
                    explanation += f"\nThe {sorted_ops[0][0]} operator leads convergence "
                    explanation += "but with meaningful contributions from other operators, "
                    explanation += "indicating balanced search dynamics.\n"
                else:
                    explanation += "\nMultiple operators contribute similarly to convergence, "
                    explanation += "indicating a well-distributed search process.\n"
        
        # Cross-algorithm comparison
        top_by_algo = {}
        for algo, operators in normalized_cds.items():
            if operators:
                top_op = max(operators.items(), key=lambda x: x[1])
                top_by_algo[algo] = (top_op[0], top_op[1])
        
        explanation += "\nCOMPARATIVE CONVERGENCE DRIVER ANALYSIS:\n"
        
        # Find algorithm with the strongest dominant driver
        if top_by_algo:
            strongest_algo = max(top_by_algo.items(), key=lambda x: x[1][1])
            explanation += f"- {strongest_algo[0]} shows the strongest single-operator "
            explanation += f"dependency on {strongest_algo[1][0]} ({strongest_algo[1][1]:.3f}).\n"
        
        # Compare driver diversity across algorithms
        diversities = {}
        for algo, operators in normalized_cds.items():
            if not operators:
                continue
                
            # Use entropy as a measure of diversity
            # Higher entropy = more diverse contributions
            values = list(operators.values())
            total = sum(values)
            if total > 0:
                probs = [v/total for v in values]
                entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
                diversities[algo] = entropy
        
        if diversities:
            most_diverse = max(diversities.items(), key=lambda x: x[1])
            least_diverse = min(diversities.items(), key=lambda x: x[1])
            
            explanation += f"- {most_diverse[0]} shows the most diverse set of convergence drivers "
            explanation += f"(entropy: {most_diverse[1]:.3f}), indicating multiple mechanisms contribute.\n"
            
            explanation += f"- {least_diverse[0]} shows the least diverse driver profile "
            explanation += f"(entropy: {least_diverse[1]:.3f}), relying more heavily on fewer operators.\n"
        
        return explanation
    
    # Main function to calculate and visualize CDS
    def analyze_cds(OAM):
        """
        Perform Convergence Driver Score analysis and visualization.
        
        Args:
            OAM: Operator Attribution Matrix
        """
        print("\nCalculating Convergence Driver Scores (CDS)...")
        
        # Calculate CDS metrics
        per_iteration_cds, total_cds, normalized_cds = calculate_cds(OAM)
        
    
        
        # Generate and print explanation
        explanation = explain_cds_analysis(total_cds, normalized_cds)
        print(explanation)
        
        return per_iteration_cds, total_cds, normalized_cds
    
    # Usage function to be called after all algorithms run
    def generate_evomapx_explanation(OAM, PEG, PEGs, selected_func_name):
        """Main function to generate all explanations after algorithms complete."""
        # Package fitness curves
        fitness_curves = {
            "GA": fitness_ga,
            "PSO": fitness_pso,
            "CS": fitness_cs,
            "DE": fitness_de
        }
        
        # Print operator effectiveness analysis for each algorithm
        print("\n=== Genetic Algorithm Operator Analysis ===")
        analyze_algorithm_operators(OAM["GA"])
        
        print("\n=== Particle Swarm Optimization Operator Analysis ===")
        analyze_algorithm_operators(OAM["PSO"])
        
        print("\n=== Cuckoo Search Operator Analysis ===")
        analyze_algorithm_operators(OAM["CS"])
        
        print("\n=== Differential Evolution Operator Analysis ===")
        analyze_algorithm_operators(OAM["DE"])
        
        # Calculate and visualize Convergence Driver Scores (CDS)
        per_iteration_cds, total_cds, normalized_cds = analyze_cds(OAM)
        
        # Generate comprehensive explanations
        generate_comparative_explanations(OAM, fitness_curves, selected_func_name, PEG, PEGs)
        
        # Print final cross-algorithm operator effectiveness comparison
        print("\n=== Cross-Algorithm Operator Effectiveness Comparison ===")
        print_operator_analysis()
    
    # Add this call at the end of your main code
    generate_evomapx_explanation(OAM, PEG, PEGs, selected_func_name)

finally:
    # Restore stdout and close the text file
    sys.stdout = original_stdout
    text_file.close()
    print(f"Text output saved to {txt_filename}")

# Now convert the text file to PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("helvetica", size=12)

# Open and read the text file
with open(txt_filename, 'r', encoding='utf-8') as f:
    for line in f:
        # Handle potential Unicode issues by replacing problematic characters
        safe_line = line.encode('latin-1', errors='replace').decode('latin-1')
        pdf.write(8, safe_line)

pdf.output(pdf_filename)
print(f"PDF output saved to {pdf_filename}")