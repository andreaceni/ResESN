# Residual Echo State Network (ResESN)

This repository provides the official implementations and experiments in the paper 
```
Ceni, Andrea, and Claudio Gallicchio. 
"Residual Echo State Networks: Residual recurrent neural networks with stable dynamics and fast learning."
Neurocomputing (2024): 127966.
```
that you can find at this link https://www.sciencedirect.com/science/article/pii/S0925231224007379.

![ResESN](/assets/ResESN.png "ResESN")

## Examples of usage on the Narma30 task

The following runs the Narma30 task for 100 different random instantiations of ResESN with 100 reservoir neurons
```
python narma30_task.py --n_hid 100 --test_trials 100 --use_test --bias_scaling 0.25 --inp_scaling 0.25 --rho 0.9 --alpha 0.9 --beta 0.2 --show_result True --solver svd --avoid_rescal_effective --cpu
```

To use the ResESN_C variant use the flag --ortho cycle as follows
```
python narma30_task.py --n_hid 100 --test_trials 100 --use_test --ortho cycle --bias_scaling 0.5 --inp_scaling 0.25 --rho 0.8 --alpha 0.9 --beta 0.4 --show_result True --solver svd --avoid_rescal_effective --cpu
```
To use the ResESN_I variant use the flag --ortho identity as follows
```
python narma30_task.py --n_hid 100 --test_trials 100 --use_test --ortho identity --bias_scaling 0.25 --inp_scaling 0.25 --rho 0.9 --alpha 0.1 --beta 1 --show_result True --solver svd --avoid_rescal_effective --cpu
```

In the file best_configurations_for_ResESN_variants.sh you can find the best settings found for each ResESN variant for each experiment in the paper.


## Citation
```
@article{ceni2024residual,
  title={Residual Echo State Networks: Residual recurrent neural networks with stable dynamics and fast learning},
  author={Ceni, Andrea and Gallicchio, Claudio},
  journal={Neurocomputing},
  pages={127966},
  year={2024},
  publisher={Elsevier}
}
```
