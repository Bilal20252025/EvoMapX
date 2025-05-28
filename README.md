# EvoMapX: a novel explainable AI framework designed to interpret the internal dynamics of population-based optimization algorithms

## Overview
EvoMapX is a sophisticated evolutionary algorithm framework that combines multiple optimization algorithms (GA, PSO, CS, DE) with advanced visualization and analysis capabilities. It specializes in solving CEC2021 benchmark problems while providing detailed insights into algorithm behavior through various analytical tools.

## Key Features
- **Multiple Optimization Algorithms**:
  - Genetic Algorithm (GA)
  - Particle Swarm Optimization (PSO)
  - Cuckoo Search (CS)
  - Differential Evolution (DE)

- **Advanced Analysis Tools**:
  - Operator Attribution Matrix (OAM
  - Population Evolution Graph (PEG)
  - Convergence Driver Score (CDS)

- **High-Quality Visualizations**:
  - Publication-ready figures (600 DPI)
  - Sharp, bold axes and lines
  - Professional typography with Times New Roman font
  - Comprehensive plotting functions for algorithm analysis

- **Robust Framework Features**:
  - Latin Hypercube Sampling for population initialization
  - Adaptive tournament selection
  - Comprehensive diversity management
  - Dual logging system (console and file output)

## Requirements
```
python>=3.8
numpy
matplotlib
networkx
seaborn
fpdf
pandas
scipy
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/evomapx.git
cd evomapx
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the main optimization framework:
```bash
python "Version 20 EvoMapX-based CEC2021 (all).py"
```

2. Select the desired CEC2021 benchmark function when prompted

3. The framework will:
   - Initialize population using LHS
   - Execute the selected optimization algorithm
   - Generate analysis visualizations
   - Save results to timestamped output files

## Output
- Optimization results and analysis are saved in both PDF and TXT formats
- Visualization files include:
  - Convergence plots
  - OAM heatmaps
  - PEG network diagrams
  - CDS analysis charts

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use EvoMapX in your research, please cite:
EvoMapX: An Explainable AI Framework for Metaheuristic Optimization Algorithms

## Contact
Corresponding author: 
Bilal H. Abed-alguni   
Bilal.h@yu.edu.jo          
Department of Computer Sciences, Yarmouk University, Irbid, Jordan
