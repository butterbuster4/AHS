# Puzzle Evolutionary Solver

This project implements a Hierarchical Solver For Reassembling Mixed Puzzles Of Eroded Gaps for jigsaw puzzles.  

---
## 🚀 Features

- **Genetic Algorithm Framework**
  - Individuals represented as groups of tiles
  - Crossover strategies with shared/buddy/best-match priority
  - Mutation operators for diversity

- **Neural Network Integration**
  - `ComboNet` (based on EfficientNet) used for tile compatibility prediction
  - Supports fine-tuning for difficult samples

- **Evaluation**
  - Piece-level and group-level reconstruction accuracy
  - Customizable metrics

- **Modular Design**
  - Easy to extend with new crossover, mutation, and fitness functions

---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/puzzle-evolution-solver.git
   cd AHS
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📚 Usage

### 🔎 Solving a Puzzle
To solve a puzzle, place your input images in the `images/` folder and run the solver:
```bash
python solve.py
```

### Customizing the Solver
### ⚙️ Customizing the Solver

You can adjust the solver’s behavior by modifying the parameters in `solve.py`:

```python
MODEL_PATH_HORI = './models/hori_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth'  
MODEL_PATH_VRTI = './models/vrti_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth'  
MODEL_PATH_PEF = './models/best_efficientnetb3.pth'  

PUZZLE_NUMBER = 2                 # Number of images to select for the puzzle  
PUZZLE_PIECE_NUMBER = 9           # Total number of pieces (e.g., 3x3 grid)  
TILE_PER_ROW = 3                  # Number of tiles per row  
TILE_EFFECTIVE_SIZE = 96          # Effective size of each tile  
GAP = 4                           # Pixel gap between tiles  
PUZZLE_IMAGE_PATH = './puzzle.png' # Path to the puzzle image  

DEFAULT_GENERATIONS = 1000        # Max generations for evolution  
DEFAULT_POPULATION = 500          # Population size  
ELITE_NUMBER = 5                  # Number of elites kept each generation  
TERMINATION_THRESHOLD = 10        # Stop if no improvement after N generations  
```

### 🖼️ Example
Input puzzle (2x3x3 with gaps):

![Input Puzzle](./examples/puzzle.png)

Reconstructed output:

![Solved Puzzle](./examples/best_individual.jpg)


## 📜 License
This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.
