
# Entry 8
python gade_lut_train.py --act_func 'exp'   --num_splits 7  --total_iters 500 --x_range -9.0 0.0 --sp_range -9.0 0.0 --num_runs 10 --dynamic
python gade_lut_train.py --act_func 'reci'  --num_splits 7  --total_iters 500 --x_range  1.0 2.0 --sp_range  1.0 2.0 --num_runs 10 --dynamic
python gade_lut_train.py --act_func 'rsqrt' --num_splits 7  --total_iters 500 --x_range  1.0 4.0 --sp_range  1.0 4.0 --num_runs 10 --dynamic
python gade_lut_train.py --act_func 'gelu'  --num_splits 7  --total_iters 500 --x_range -4.0 4.0 --sp_range -4.0 4.0 --num_runs 10 --dynamic
python gade_lut_train.py --act_func 'silu'  --num_splits 7  --total_iters 500 --x_range -4.0 4.0 --sp_range -4.0 4.0 --num_runs 10 --dynamic

# Entry 16
python gade_lut_train.py --act_func 'exp'   --num_splits 15 --total_iters 500 --x_range -9.0 0.0 --sp_range -9.0 0.0 --num_runs 10 --dynamic
python gade_lut_train.py --act_func 'reci'  --num_splits 15 --total_iters 500 --x_range  1.0 2.0 --sp_range  1.0 2.0 --num_runs 10 --dynamic
python gade_lut_train.py --act_func 'rsqrt' --num_splits 15 --total_iters 500 --x_range  1.0 4.0 --sp_range  1.0 4.0 --num_runs 10 --dynamic
python gade_lut_train.py --act_func 'gelu'  --num_splits 15 --total_iters 500 --x_range -4.0 4.0 --sp_range -4.0 4.0 --num_runs 10 --dynamic 
python gade_lut_train.py --act_func 'silu'  --num_splits 15 --total_iters 500 --x_range -4.0 4.0 --sp_range -4.0 4.0 --num_runs 10 --dynamic