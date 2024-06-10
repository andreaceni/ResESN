import numpy as np
import time
import argparse
#from esn import DeepReservoir
#from sklearn import preprocessing
#from sklearn.linear_model import Ridge
#import torch.nn.utils
from myESN import ESN


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=100,
                    help='hidden size of recurrent net')
#parser.add_argument('--cpu', action="store_true")
parser.add_argument('--inp_scaling', type=float, default=0.1,
                    help='ESN input scaling')
parser.add_argument('--rho', type=float, default=0.9,
                    help='ESN spectral radius')
parser.add_argument('--leaky', type=float, default=1.0,
                    help='ESN leakage')
parser.add_argument('--alpha', type=float, default=0.0,
                    help='ResESN alpha')
parser.add_argument('--beta', type=float, default=1.0,
                    help='ResESN beta')
parser.add_argument('--regul', type=float, default=0.0,
                    help='Ridge regularisation parameter')
parser.add_argument('--tot_trials', type=int, default=10)
parser.add_argument('--show_result', action="store_true",
                    help='print results for each delay')
parser.add_argument('--bias',  action="store_true",         # bias harms MC
                    help='bias inside the nonlinearity') 
#parser.add_argument('--norm_acts',  action="store_true",
#                    help='normalise activations before Ridge regression') 
parser.add_argument('--distrib', type=str, default='normal') # it can be 'uniform' or 'normal'
parser.add_argument('--nonlin', type=str, default='tanh') # it can be 'tanh' 'identity' 'sigmoid'
parser.add_argument('--simple_cycle', action="store_true",
                    help='simple cycle reservoir topology')
parser.add_argument('--ortho', type=str, default='random') # it can be 'random' 'cycle' 'identity'
parser.add_argument('--Euler', action="store_true",
                    help='implement EuESN architecture')
parser.add_argument('--eu_step', type=float, default=1.0,
                    help='Euler step to run the EuESN')
parser.add_argument('--diffus', type=float, default=1.0,
                    help='diffusione term to run the EuESN')
parser.add_argument('--EuRec_scal', type=float, default=1.0,
                    help='scaling term to generate recurrent matrix of EuESN')
parser.add_argument('--nonlinMemory', action="store_true",
                    help='reconstruct sinusoidal delayed input')

main_folder = 'result'

args = parser.parse_args()
print(args)

if args.alpha == 0: # ResESN
    namefile = 'MemoryCapacity_log_ESN'
else: # ESN
    if args.ortho == 'random':
        namefile = 'MemoryCapacity_log_ResESN'
    elif args.ortho == 'cycle':
        namefile = 'MemoryCapacity_log_ResESN_C'
    elif args.ortho == 'identity':
        namefile = 'MemoryCapacity_log_ResESN_I'
    else:
        raise ValueError('Only random, cycle, and identity, are supported for the orthogonal branch.')

if args.simple_cycle:
    namefile += '_SCR'

f = open(f'{main_folder}/{namefile}.txt', 'a')
ar = ''
for k, v in vars(args).items():
    ar += f'{str(k)}: {str(v)}, '
f.write(ar + '\n')
f.write('**************\n')
f.close()

#device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
#print("Using device ", device)
n_inputs = 1
n_outputs = 1
tot_trials = args.tot_trials
max_delay = 2*args.n_hid
if args.bias:
    str_bias = 'random'
else:
    str_bias = 'null'
if args.distrib == 'uniform':
    reservoir_uniform = True
else:
    reservoir_uniform = False

MC_standard = np.zeros(tot_trials)

for guess in range(tot_trials):

    iniz=time.time()

    # ############################ initialise reservoir ############################ #
    seed = np.random.randint(low=1, high=10000)
    #seed = 1881
    str_trial_seed = 'Trial n. ' + str(guess+1) + '.    Seed: ' + str(seed) + '\n'
    print(str_trial_seed)
    f = open(f'{main_folder}/{namefile}.txt', 'a')
    f.write(str_trial_seed)
    f.close()
    if args.Euler:
        esn = ESN(n_inputs=n_inputs,
                n_outputs=n_outputs,
                nonlinearity = args.nonlin,
                n_reservoir= args.n_hid, #
                spectral_radius= args.rho , # 
                sparsity= 0. ,
                noise= 0*1e-6, 
                input_scaling= args.inp_scaling,
                feedback = False ,
                states_pow2 = False,
                teacher_scaling= 1. ,
                transient=100,
                leak_rate= args.leaky , #
                bias = str_bias,
                Euler=True, eu_step=args.eu_step, diffus=args.diffus, rec_scal=args.EuRec_scal, # implement EuSN
                reservoir_uniform=reservoir_uniform,
                random_state=seed)
    else:
        esn = ESN(n_inputs=n_inputs,
                n_outputs=n_outputs,
                nonlinearity = args.nonlin,
                n_reservoir= args.n_hid, #
                spectral_radius= args.rho , # 
                sparsity= 0. ,
                noise= 0*1e-6, 
                input_scaling= args.inp_scaling,
                feedback = False ,
                states_pow2 = False,
                teacher_scaling= 1. ,
                transient=100,
                leak_rate= args.leaky , #
                bias = str_bias,
                reservoir_uniform=reservoir_uniform,
                random_state=seed)

        # Linear Orthogonal Branch
        if args.alpha != 0:
            dims= esn.n_reservoir
            alphas = args.alpha * np.ones(dims)
            betas = args.beta * np.ones(dims)
            if args.ortho == 'random':
                # get a random orthogonal matrix drawn from the O(N) Haar distribution (the only uniform distribution on O(N)).
                #from scipy.stats import ortho_group  # Requires version 0.18 of scipy
                #Ortog = ortho_group.rvs(dim=dims)
                # get a random orthogonal matrix from QR decomposition of a random uniform matrix in [-1,1)
                Ortog, _ = np.linalg.qr(2*np.random.rand(dims,dims)-1)
            elif args.ortho == 'cycle':
                # SimpleCycle 
                Ortog = np.zeros((args.n_hid, args.n_hid))
                Ortog[0, args.n_hid-1] = 1
                for i in range(args.n_hid-1):
                    Ortog[i+1,i] = 1  
            elif args.ortho == 'identity':
                Ortog = np.eye(args.n_hid)
            else:
                raise ValueError('Only random, cycle, and identity, are supported for the orthogonal branch.')
            esn.OLESN(alphas, betas, Ortog )

        if args.simple_cycle:
            # SimpleCycle reservoir 
            W = np.zeros((args.n_hid, args.n_hid))
            W[0, args.n_hid-1] = args.rho
            for i in range(args.n_hid-1):
                W[i+1,i] = args.rho  # Tino used 0.5 but with input in [-0.5,0.5] 
            esn.W_res = W
    # ############################################################################### #
    
    MC_original = np.zeros(max_delay+1)
    for k in range(1,max_delay+1):
        print('----------------------------------------------------------------------------------------------------')
        f = open(f'{main_folder}/{namefile}.txt', 'a')
        f.write('----------------------------------------------------------------------------------------------------\n')
        f.close()

        # ############# generate random uniform time series data ############### #
        length = 6000
        time_series = np.random.uniform(-0.8, 0.8, length+k) # [-1,1] with Legendre polynomials? I mean, they are still orthogonal in [-0.8, 0.8]
        X_data = np.zeros(length)
        X_data = time_series[k:length+k]

        train_length = 5000
        trX = X_data[:train_length]
        input_train = trX[:train_length]
        test_length = 1000
        tsX = X_data[train_length:]
        input_test = tsX[:test_length]
        # ###################################################################### #

        k_delay = k
        if args.nonlinMemory:
            target_train = np.sin(np.pi * time_series[:train_length])
            target_test = np.sin(np.pi * time_series[train_length:-k_delay])
        else:
            target_train = time_series[:train_length]
            target_test = time_series[train_length:-k_delay]

        # Set optimisation method
        esn.optim(opt='ridge_regr', teacher_forcing=False)

        ################## FIT ##################
        train_predicted = esn.fit(input_train, target_train, esn.W_out)

        ################## PREDICT ###################
        test_predicted, test_InternalStates = esn.predict(input_test, noise_closloop=False, continuation=True) # TEST 
        test_pred = test_predicted[:,0]

        # original
        target_mean = np.mean(target_test)
        output_mean = np.mean(test_pred) 
        num  = 0
        denom_t = 0
        denom_out = 0
        for i in range(test_length):
            deviat_t = target_test[i] - target_mean
            deviat_out = test_pred[i] - output_mean
            num += deviat_t * deviat_out
            denom_t += deviat_t**2
            denom_out += deviat_out**2
        num = num**2
        den = denom_t * denom_out
        MC_original[k] = num/den

        str_MC = 'Delay: ' +str(k) + ',        MC_original: ' + str(MC_original[k]) + '\n'
        if args.show_result:
            print(str_MC)
        f = open(f'{main_folder}/{namefile}.txt', 'a')
        f.write(str_MC)
        f.close()

    mc_original = sum(MC_original)
    MC_standard[guess] = mc_original
    fin=time.time()

    final_str_MC = ''
    for i in range(5):
        final_str_MC += '----------------------------------------------------------------------------------------------------\n'
    final_str_MC += 'MEMORY CAPACITY.        Original: ' + str(mc_original) + '\n'
    final_str_MC += 'Computational time: ' + str(fin-iniz) + 's\n'
    for i in range(6):
        final_str_MC += '|\n'
    print(final_str_MC)
    f = open(f'{main_folder}/{namefile}.txt', 'a')
    f.write(final_str_MC)
    f.close()


meanMC = np.mean(MC_standard)
stdMC = np.std(MC_standard)
result_MC = ' ##################################################################### \n'
result_MC += 'Mean MC: ' + str(meanMC) + ',  std ' + str(stdMC) + '\n'
result_MC += ' ##################################################################### \n'
for i in range(6):
    result_MC += '#\n'
print(result_MC)
f = open(f'{main_folder}/{namefile}.txt', 'a')
f.write(result_MC)
f.close()