# LUT-APP: Dynamic-Precision LUT-based Approximation Unifying Non-Linear Operations in Transformers
The official PyTorch Implementation of Genetic Adaptive Differentail Evolution(GADE) in "LUT-APP: Dynamic-Precision LUT-based Approximation Unifying Non-Linear Operations in Transformers" [DATE 2026]

<p align="center"><img width="1343" height="454" alt="DFF_Figure_HR_whift_back" src="https://github.com/user-attachments/assets/3211f8b2-fc0c-4e9f-a222-629ecd71f5e7" /></p>
<p align="center"><img width="750" height="600" alt="HW_Figure_rev_final_rev_final_white" src="https://github.com/user-attachments/assets/cdd65700-0a63-486c-b424-89f542a19444" /></p>

## Installation

Clone this repo:

```
git clone https://github.com/IDSL-SeoulTech/LUT-APP.git
cd LUT-APP/
```

The code is implemented with Python > 3.9, PyTorch > 1.8.
It is recommended to use Anaconda for making environments required for this code.

Create an anaconda environment:
```
conda env create -f requirement.yml
conda activate gade_env
```

## How to Use

### Supporting Non-linear Opeartions List

```
Exponential (EXP)
Reciprocal (RECI)
Reciprocal Root Squre Root (RSQRT)
GeLU (GeLU)
SiLU (SiLU)
```

### Algorithm Script

```
python gade_lut_train.py --act_func (non-linear function name) --num_splits (segments - 1) --total_iters (# of iterations) --x_range (Input Range) --sp_range (Breakpoint Range) --num_runs (# of runs) --dynamic
```

Example of approximationg EXP with 8 segments:

```
python gade_lut_train.py --act_func 'exp' --num_splits 7 --total_iters 500 --x_range -9.0 0.0 --sp_range -8.0 0.0 --num_runs 10 --dynamic
```

### Evaluation Script

```
python operator_val.py --act_func (non-linear function name) --distance (distance between samples) --input_range (Input Range) --dynamic --param_path (Path to PWL Perameter)
```

Example of evaluating EXP:

```
python operator_val.py --act_func 'exp' --distance 0.0009765625 --input_range -9.0 0 --dynamic --param_path ./gade_pwl_param/exp/entry_8/exp_7_dff_True_seed_7202_0.001812850.json
```

### Reproduce Paper Result

```
chomod -R 777 ./gade_lut_train_prompt_dff_True.csh
./gade_lut_train_prompt_dff_True.csh
```

## References
This project is based on the following repository:
  - https://github.com/PingchengDong/GQA-LUT
    (Used as the base implementation)
