# best configurations for ResESN variants 

# narma10
python narma10_task.py --n_hid 100 --bias_scaling 1 --inp_scaling 0.25 --rho 1 --leaky 1 --alpha 0.1 --beta 0.95 --show_result True --test_trials 100  --use_test --solver svd --avoid_rescal_effective --cpu
python narma10_task.py --n_hid 100 --ortho cycle --bias_scaling 2 --inp_scaling 0.25 --rho 1.1 --leaky 1 --alpha 0.0001 --beta 1.0 --show_result True --test_trials 100  --use_test --solver svd --avoid_rescal_effective --cpu
python narma10_task.py --n_hid 100 --ortho identity --bias_scaling 1 --inp_scaling 0.25 --rho 0.9 --leaky 1 --alpha 0.01 --beta 1.05 --show_result True --test_trials 100  --use_test --solver svd --avoid_rescal_effective --cpu

# narma30
python narma30_task.py --n_hid 100 --test_trials 100  --use_test --bias_scaling 0.25 --inp_scaling 0.25 --rho 0.9 --alpha 0.9 --beta 0.2 --show_result True --solver svd --avoid_rescal_effective --cpu
python narma30_task.py --n_hid 100 --test_trials 100  --use_test --ortho cycle --bias_scaling 0.5 --inp_scaling 0.25 --rho 0.8 --alpha 0.9 --beta 0.4 --show_result True --solver svd --avoid_rescal_effective --cpu
python narma30_task.py --n_hid 100 --test_trials 100  --use_test --ortho identity --bias_scaling 0.25 --inp_scaling 0.25 --rho 0.9 --leaky 1 --alpha 0.1 --beta 1 --show_result True --solver svd --avoid_rescal_effective --cpu

# sinMem
python piSineDelay10_task.py --ergodic --test_trials 10  --use_test --n_hid 100 --bias_scaling 0 --inp_scaling 2 --rho 0.6 --alpha 0.6 --beta 0.0001 --show_result True --solver svd --avoid_rescal_effective --cpu
python piSineDelay10_task.py --ergodic --test_trials 10  --use_test --n_hid 100 --ortho cycle --bias_scaling 0 --inp_scaling 2 --rho 0.5 --leaky 1 --alpha 0.6 --beta 0.0001 --show_result True --solver svd --avoid_rescal_effective --cpu
python piSineDelay10_task.py --ergodic --test_trials 100  --use_test --n_hid 100 --ortho identity --bias_scaling 0 --inp_scaling 0.5 --rho 0.9 --leaky 1 --alpha 0.01 --beta 0.95 --show_result True --solver svd --avoid_rescal_effective --cpu

# ctXOR
python ctXOR_task.py --test_trials 100 --use_test --delay 5 --n_hid 100 --bias_scaling 2 --inp_scaling 0.25 --rho 0.5 --leaky 1 --alpha 0.4 --beta 0.4 --show_result True --solver svd --avoid_rescal_effective --cpu
python ctXOR_task.py --test_trials 100 --use_test --delay 5 --n_hid 100 --ortho cycle --bias_scaling 2 --inp_scaling 0.25 --rho 0.7 --leaky 1 --alpha 0.4 --beta 0.2 --show_result True --solver svd --avoid_rescal_effective --cpu
python ctXOR_task.py --test_trials 100 --use_test --delay 5 --n_hid 100 --ortho identity --bias_scaling 0.5 --inp_scaling 0.25 --rho 0.5 --leaky 1 --alpha 0 --beta 0.4 --show_result True --solver svd --avoid_rescal_effective --cpu

# mg
python mackey-glass_task.py --n_hid 100 --bias_scaling 2 --inp_scaling 2 --rho 0.6 --alpha 0.8 --beta 0.8 --show_result True --solver svd --avoid_rescal_effective --cpu --test_trials 100  --use_test
python mackey-glass_task.py --n_hid 100 --ortho cycle --bias_scaling 1 --inp_scaling 0.5 --rho 1.1 --alpha 0.1 --beta 1.01 --show_result True --solver svd --avoid_rescal_effective --cpu --test_trials 100  --use_test
python mackey-glass_task.py --n_hid 100 --ortho identity --bias_scaling 0.25 --inp_scaling 0.5 --rho 0.9 --leaky 1 --alpha 0.1 --beta 0.9 --show_result True --solver svd --avoid_rescal_effective --cpu --test_trials 100  --use_test

# 84mg
python mackey-glass_task.py --lag 84 --test_trials 100  --use_test --n_hid 100 --bias_scaling 0.5 --inp_scaling 1 --rho 1.1 --alpha 0.0001 --beta 0.99 --show_result True --solver svd --avoid_rescal_effective --cpu 
python mackey-glass_task.py --lag 84 --n_hid 100 --ortho cycle --bias_scaling 0.5 --inp_scaling 1 --rho 1 --alpha 0.6 --beta 0.99 --show_result True --solver svd --avoid_rescal_effective --cpu --test_trials 100  --use_test
python mackey-glass_task.py --lag 84 --n_hid 100 --ortho identity --bias_scaling 1 --inp_scaling 2 --rho 1 --alpha 0.8 --beta 0.2 --show_result True --solver svd --avoid_rescal_effective --cpu --test_trials 100  --use_test

# memory capacity
python myMC_task.py --n_hid 100 --show_result --rho 0.9 --inp_scaling 0.25 --alpha 0.95 --beta 0.01 --tot_trials 10
python myMC_task.py --n_hid 100 --ortho cycle --show_result --rho 0.9 --inp_scaling 0.25 --alpha 0.95 --beta 0.0001 --tot_trials 10
python myMC_task.py --n_hid 100 --ortho identity --show_result --rho 0.9 --inp_scaling 0.25 --alpha 0.01 --beta 1.1 --tot_trials 10

# 25Lz
python lorenz_task.py --cpu --test_trials 10 --use_test --serieslen 2 --lag 25 --n_hid 100 --bias_scaling 1 --inp_scaling 0.25 --rho 1.1 --leaky 1 --alpha 0.6 --beta 0.01 --show_result True --solver svd --avoid_rescal_effective
python lorenz_task.py --cpu --test_trials 10 --use_test --serieslen 2 --lag 25 --n_hid 100 --ortho cycle --bias_scaling 2 --inp_scaling 0.25 --rho 0.6 --leaky 1 --alpha 0.0001 --beta 0.6 --show_result True --solver svd --avoid_rescal_effective
python lorenz_task.py --cpu --test_trials 10 --use_test --serieslen 2 --lag 25 --n_hid 100 --ortho identity --bias_scaling 1 --inp_scaling 0.25 --rho 1.1 --leaky 1 --alpha 0.0001 --beta 0.001 --show_result True --solver svd --avoid_rescal_effective


# psmnist
CUDA_VISIBLE_DEVICES=0 python testing_psMNIST.py --regul 0 --solver svd --test_trials 1 --use_test --batch_train 32 --batch_test 50 --n_hid 16384 --bias_scaling 2 --inp_scaling 0.25 --rho 0.9 --leaky 1 --alpha 0.998 --beta 0.05 --show_result True --avoid_rescal_effective
CUDA_VISIBLE_DEVICES=0 python testing_psMNIST.py --regul 0 --solver svd --ortho cycle --test_trials 1 --use_test --batch_train 32 --batch_test 50 --n_hid 16384 --bias_scaling 2 --inp_scaling 0.25 --rho 0.9 --leaky 1 --alpha 0.998 --beta 0.05 --show_result True --avoid_rescal_effective
CUDA_VISIBLE_DEVICES=0 python testing_psMNIST.py --regul 0 --solver svd --ortho identity --test_trials 1 --use_test --batch_train 32 --batch_test 50 --n_hid 16384 --bias_scaling 0.1 --inp_scaling 0.1 --rho 1 --leaky 1 --alpha 0.002 --beta 0.99 --show_result True --avoid_rescal_effective
