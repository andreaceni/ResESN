import numpy as np
import torch.nn.utils
import argparse
from esn import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from utils import get_narma10


parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--n_hid', type=int, default=256,
                    help='hidden size of recurrent net')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--inp_scaling', type=float, default=1.,
                    help='ESN input scaling')
parser.add_argument('--rho', type=float, default=0.99,
                    help='ESN spectral radius')
parser.add_argument('--leaky', type=float, default=1.0,
                    help='ESN leakage')
parser.add_argument('--alpha', type=float, default=0.0,
                    help='ResESN alpha')
parser.add_argument('--beta', type=float, default=1.0,
                    help='ResESN beta')
parser.add_argument('--regul', type=float, default=0.0,
                    help='Ridge regularisation parameter')
parser.add_argument('--lag', type=int, default=1)
parser.add_argument('--use_test', action="store_true")
parser.add_argument('--show_result', type=bool, default=False)
parser.add_argument('--test_trials', type=int, default=1,
                    help='number of trials to compute mean and std on test')
parser.add_argument('--ortho', type=str, default='random') # it can be 'random' 'cycle' 'identity' 'null'
parser.add_argument('--simple_cycle', action="store_true",
                    help='simple cycle reservoir topology')
#
parser.add_argument('--bias_scaling', type=float, default=None,
                    help='ESN bias scaling')
parser.add_argument('--avoid_rescal_effective', action="store_false")
parser.add_argument('--solver', type=str, default=None,
                    help='Ridge ScikitLearn solver')
parser.add_argument('--Euler', action="store_true",
                    help='implement EuESN architecture')
parser.add_argument('--eu_step', type=float, default=1.0,
                    help='Euler step to run the EuESN')
parser.add_argument('--diffus', type=float, default=1.0,
                    help='diffusione term to run the EuESN')
parser.add_argument('--EuRec_scal', type=float, default=1.0,
                    help='scaling term to generate recurrent matrix of EuESN')
parser.add_argument('--ES2N', action="store_true",
                    help='implement ES2N architecture')
                    

args = parser.parse_args()
print(args)
if args.ortho == 'random':
    namefile = 'narma10_log_ResESN'
elif args.ortho == 'cycle':
    namefile = 'narma10_log_ResESN_C'
elif args.ortho == 'identity':
    namefile = 'narma10_log_ResESN_I'
elif args.ortho == 'null':
    namefile = 'narma10_log_ESN'
else:
    raise ValueError('Only random, cycle, identity, and null, are supported for the orthogonal branch.')

if args.simple_cycle:
    namefile += '_SCR'

if args.Euler:
    namefile = 'narma10_log_EuESN'

if args.ES2N:
    namefile = 'narma10_log_ES2N'
    args.beta = 1 - args.alpha

main_folder = 'result'

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print("Using device ", device)
n_inp = 1
n_out = 1
washout = 200
lag = args.lag

(train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_narma10(args.lag)


NRMSE = np.zeros(args.test_trials)
for guess in range(args.test_trials):

    if args.Euler:
        model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho,
                                input_scaling=args.inp_scaling,
                                connectivity_recurrent=args.n_hid,
                                connectivity_input=args.n_hid, 
                                leaky=args.leaky,
                                alpha=args.alpha,
                                beta=args.beta,
                                #
                                effective_rescaling=args.avoid_rescal_effective,
                                bias_scaling=args.bias_scaling,
                                Euler=True,
                                epsilon=args.eu_step, 
                                gamma=args.diffus,
                                recur_scaling=args.EuRec_scal,
                                ).to(device)
    else:
        model = DeepReservoir(n_inp, tot_units=args.n_hid, spectral_radius=args.rho,
                                input_scaling=args.inp_scaling,
                                connectivity_recurrent=args.n_hid,
                                connectivity_input=args.n_hid, 
                                leaky=args.leaky,
                                alpha=args.alpha,
                                beta=args.beta,
                                #
                                effective_rescaling=args.avoid_rescal_effective,
                                bias_scaling=args.bias_scaling,
                                ).to(device)

    # ########### define the orthogonal branch ########### #
    if args.ortho == 'random':
        Ortog, _ = np.linalg.qr(2*np.random.rand(args.n_hid,args.n_hid)-1)
    elif args.ortho == 'cycle':
        # SimpleCycle 
        Ortog = np.zeros((args.n_hid, args.n_hid))
        Ortog[0, args.n_hid-1] = 1
        for i in range(args.n_hid-1):
            Ortog[i+1,i] = 1  
    elif args.ortho == 'identity':
        Ortog = np.eye(args.n_hid)
    elif args.ortho == 'null':
        Ortog = np.zeros(args.n_hid)
    else:
        raise ValueError('Only random, cycle, identity, and null, are supported for the orthogonal branch.')
        
    model.reservoir[0].net.ortho = torch.nn.Parameter(torch.Tensor(Ortog).to(device), requires_grad=False)
    # #################################################### #

    if args.simple_cycle:
        # SimpleCycle reservoir 
        W = np.zeros((args.n_hid, args.n_hid))
        W[0, args.n_hid-1] = args.rho
        for i in range(args.n_hid-1):
            W[i+1,i] = args.rho  # Tino used 0.5 but with input in [-0.5,0.5] 
        model.reservoir[0].net.recurrent_kernel = torch.nn.Parameter(torch.Tensor(W).to(device), requires_grad=False)


    #(train_dataset, train_target), (valid_dataset, valid_target), (test_dataset, test_target) = get_narma10(args.lag)

    @torch.no_grad()
    def test_esn(dataset, target, classifier, scaler):
        dataset = dataset.reshape(1, -1, 1).to(device)
        target = target.reshape(-1, 1).numpy()
        activations = model(dataset)[0].cpu().numpy()
        activations = activations[:, washout:]
        activations = activations.reshape(-1, args.n_hid)
        activations = scaler.transform(activations)
        predictions = classifier.predict(activations)
        mse = np.mean(np.square(predictions - target))
        rmse = np.sqrt(mse)
        norm = np.sqrt(np.square(target).mean())
        nrmse = rmse / (norm + 1e-9)
        return nrmse

    dataset = train_dataset.reshape(1, -1, 1).to(device)
    target = train_target.reshape(-1, 1).numpy()
    activations = model(dataset)[0].cpu().numpy()
    activations = activations[:, washout:]
    activations = activations.reshape(-1, args.n_hid)
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    if args.solver is None:
        classifier = Ridge(alpha=args.regul, max_iter=1000).fit(activations, target)
    elif args.solver == 'svd':
        classifier = Ridge(alpha=args.regul, solver='svd').fit(activations, target)
    else:
        classifier = Ridge(alpha=args.regul, solver=args.solver).fit(activations, target)
    valid_nmse = test_esn(valid_dataset, valid_target, classifier, scaler)
    test_nmse = test_esn(test_dataset, test_target, classifier, scaler) if args.use_test else 0.0
    NRMSE[guess] = test_nmse


    f = open(f'{main_folder}/{namefile}.txt', 'a')
    ar = ''
    for k, v in vars(args).items():
        ar += f'{str(k)}: {str(v)}, '
    ar += f'valid: {str(round(valid_nmse, 5))}, test: {str(round(test_nmse, 5))}'
    f.write(ar + '\n')
    f.write('**************\n\n\n')
    f.close()

    if args.show_result:
        print(ar)


mean = np.mean(NRMSE)
std = np.std(NRMSE)
lastprint = ' ##################################################################### \n'
lastprint += 'Mean NRMSE ' + str(mean) + ',    std ' + str(std) + '\n'
lastprint += ' ##################################################################### \n'
print(lastprint)

f = open(f'{main_folder}/{namefile}.txt', 'a')
f.write(lastprint)
f.close()