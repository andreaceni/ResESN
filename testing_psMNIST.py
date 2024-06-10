import numpy as np
import torch.nn.utils
import argparse
from tqdm import tqdm
from esn import DeepReservoir
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from utils import get_mnist_testing_data


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
parser.add_argument('--regul', type=float, default=1.0,
                    help='regularisation parameter')
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
#
parser.add_argument('--batch_train', type=int, default=256,
                    help='batch size for training dataset')
parser.add_argument('--batch_test', type=int, default=200, # must be multiple of 10
                    help='batch size for validation/test dataset') 
#parser.add_argument('--permutation', type=list, default=None,
#                    help='permutation of psMNIST')                                     
                    

args = parser.parse_args()
print(args)
if args.ortho == 'random':
    namefile = 'testing_psMNIST_log_ResESN'
elif args.ortho == 'cycle':
    namefile = 'testing_psMNIST_log_ResESN_C'
elif args.ortho == 'identity':
    namefile = 'testing_psMNIST_log_ResESN_I'
elif args.ortho == 'null':
    namefile = 'testing_psMNIST_log_ESN'
else:
    raise ValueError('Only random, cycle, identity, and null, are supported for the orthogonal branch.')

if args.simple_cycle:
    namefile += '_SCR'

if args.Euler:
    namefile = 'testing_psMNIST_log_EuESN'

if args.ES2N:
    namefile = 'testing_psMNIST_log_ES2N'
    args.beta = 1 - args.alpha


main_folder = 'result'

device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
print("Using device ", device)


n_inp = 1
#n_out = 10
train_loader, test_loader = get_mnist_testing_data(args.batch_train, args.batch_test)

# fix a random permutation for the experiment
permutation = [ 60, 361, 167, 578, 107, 772, 313, 626,  81, 367, 711, 728, 348, 523,
        352, 410, 616, 421, 218, 472, 554, 520, 405, 440, 608, 444, 495,  58,
        169,  34, 525, 652, 195,  17, 779, 704, 358, 675,  29, 337, 306, 666,
        756, 512, 615, 186,  38,  12, 574, 635, 378, 256,   8, 305, 415, 285,
        295, 403, 222, 686, 575, 518, 245, 226, 599, 654,  65, 605, 560, 726,
        147,  73, 766,   4,  31,  10, 643, 109, 445,  66, 714, 397, 240,  35,
        153, 570, 259, 531, 587, 567,  49,   7, 394, 267, 359, 456, 670, 566,
        389, 321, 105, 708, 460, 505, 108, 741,  14, 272, 707, 342, 431, 297,
        468,  70, 258, 427, 719, 148, 209, 633, 671, 464, 197, 455, 137, 775,
        379, 749, 281,  62, 350, 278, 119, 653, 418, 371, 762,  92, 538,  61,
        301, 731, 298, 658,  67, 118, 691, 347, 713, 263, 540, 122,  40,  56,
        198, 519, 363, 777, 346, 557, 530, 343, 221, 579, 323,  11, 496, 442,
        135, 314, 545, 625, 291, 293,  26, 289, 433, 422, 368, 344, 411, 577,
        447, 163, 146, 141, 312, 172, 296, 667, 585, 398, 522, 205, 216, 219,
        480, 469, 536, 618, 365, 747, 439,  76,  18, 738, 164, 561, 696, 158,
        559, 384, 664, 299, 466, 609, 564, 262, 572,  48, 682, 149, 136, 244,
         78, 602, 354, 154, 100, 130, 190, 748, 248, 639, 326, 265, 171, 659,
        255, 482, 231, 509, 650, 161, 175,  94, 437, 765, 736, 481, 601, 769,
        489,  51, 477, 129, 133, 513, 500, 429, 771, 507,  83, 600,  36, 434,
        689, 391, 555, 759, 581, 742, 174, 642, 584, 548,  30, 516,   9, 700,
        479, 494, 128, 699, 264, 302, 604, 491, 621, 284, 241,  32,  21, 254,
        396, 376, 242, 181, 683, 678,  22, 199, 645, 125, 370, 200, 268, 685,
        651, 629, 735, 145, 556, 485, 210, 188, 483, 332, 300, 594, 425, 589,
        373, 582, 534, 539, 336, 493, 202, 569, 319, 157, 117, 311, 276,  53,
          6, 189, 271, 744, 474,  42, 679, 274, 760, 543,  15, 612, 783, 640,
        247, 611, 357, 730,  79,  85,  24, 535, 203, 476, 568,  68, 458, 532,
        580, 722, 206, 392, 331, 246, 183, 279,  87,  45, 646, 562, 170, 680,
        428, 406, 721, 124, 503, 644, 453,  95, 662, 768, 178, 782, 452, 330,
        619, 617, 251, 138, 399,  39, 114, 388,  69, 517, 546, 140, 372, 614,
         84, 116, 717, 674, 180,  25, 677, 432, 732, 767, 103, 637, 773, 338,
        270,  59, 492, 335, 463, 624, 764, 627,  20, 353, 590, 150, 755, 550,
        341,  72, 131, 753, 229, 603, 487, 470, 303, 374, 511, 710, 287, 734,
        641, 478,  80, 596, 177, 676, 692, 781, 329, 698, 450, 185, 690, 196,
        282, 223, 156, 526, 591, 454, 647, 681, 668, 381, 565, 235, 724, 770,
        715, 401, 307, 402, 191, 549, 176, 387, 746,  57, 598, 459, 673, 649,
        126, 448, 309,  77, 260, 529, 351, 663, 269, 165,  52, 143,  91, 310,
         93, 745, 327, 475, 423, 443, 438, 737, 606, 340, 355, 112,   5, 544,
        364,  75, 261,  16, 322, 622, 733, 409, 318, 424, 212, 375,  82, 288,
        277, 366, 227, 152,  54, 462, 628,  71, 239, 334, 697, 317, 250, 283,
        552, 217, 576, 695, 623, 757, 339, 752, 316, 106, 228, 395, 501, 187,
        151, 515, 369, 687, 390, 497, 252, 417, 542, 400, 115, 630, 607, 761,
        502, 729, 132, 362, 583, 750, 159, 776, 778, 571, 471, 111,  90, 537,
        514, 213, 356, 168, 113,  86, 720, 407, 162, 345, 660, 506, 430, 709,
        142, 743,  44,  89,  50, 412, 275, 208, 705, 657, 655, 490, 257, 104,
        665, 404, 774, 360,  64, 294, 702,  96, 484, 684, 701, 416, 382, 441,
         27, 436, 586, 558, 280,  63, 457, 498, 426, 249, 325, 573, 688, 718,
        383, 328, 524, 547, 631, 508,  46, 182,   2, 232, 385, 461, 234, 349,
        315, 380, 192, 533,   3, 499, 758, 304,  37, 740, 763, 233, 127, 739,
        238, 694, 194, 521, 563, 467, 703, 636,  43, 672, 632,  97, 634, 669,
        527, 706, 620, 184, 386, 155, 751,  13, 237, 377, 693, 712, 727, 593,
        393, 656, 273, 488, 211, 204,  55, 225, 123, 541, 166,  28,   1, 420,
         33, 193, 725, 780, 120, 102, 446, 528, 320, 465,  41, 754, 408, 597,
        207, 638, 333, 236, 308,  47,   0, 101, 419, 224, 173,  98, 230, 121,
        220,  23, 179, 716, 595,  88, 723, 413,  99, 144, 648, 324,  74,  19,
        286, 290, 613, 215, 551, 139, 449, 243, 110, 451, 201, 661, 553, 414,
        486, 292, 610, 435, 473, 214, 592, 504, 266, 160, 134, 510, 253, 588]
perm = torch.Tensor(permutation).to(torch.long).to(device)


ACC = np.zeros(args.test_trials)
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
            W[i+1,i] = args.rho  
        model.reservoir[0].net.recurrent_kernel = torch.nn.Parameter(torch.Tensor(W).to(device), requires_grad=False)
    


    @torch.no_grad()
    def test_esn(data_loader, classifier, scaler):
        activations, ys = [], []
        for x, y in tqdm(data_loader):
            x = x.to(device)
            ## Reshape images for sequence learning:
            x = x.reshape(x.shape[0], 1, 784)
            x = x.permute(0, 2, 1)
            x = x[:, perm, :]
            output = model(x)[-1][0]
            activations.append(output.cpu())
            ys.append(y)
        activations = torch.cat(activations, dim=0).numpy()
        activations = scaler.transform(activations)
        ys = torch.cat(ys, dim=0).numpy()
        return classifier.score(activations, ys)



    activations, ys = [], []
    for x, y in tqdm(train_loader):
        x = x.to(device)
        ## Reshape images for sequence learning:
        x = x.reshape(x.shape[0], 1, 784)
        x = x.permute(0, 2, 1)
        x = x[:, perm, :]
        output = model(x)[-1][0]
        activations.append(output.cpu())
        ys.append(y)
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    if args.solver is None:
        classifier = RidgeClassifier(alpha=args.regul, max_iter=1000).fit(activations, ys)
    elif args.solver == 'svd':
        classifier = RidgeClassifier(alpha=args.regul, solver='svd').fit(activations, ys)
    elif args.solver == 'logistic':
        classifier = LogisticRegression(C=args.regul, max_iter=1000).fit(activations, ys)
    else:
        classifier = RidgeClassifier(alpha=args.regul, solver=args.solver).fit(activations, ys)
    #valid_acc = test_esn(valid_loader, classifier, scaler)
    test_acc = test_esn(test_loader, classifier, scaler) if args.use_test else 0.0
    ACC[guess] = test_acc



    f = open(f'{main_folder}/{namefile}.txt', 'a')
    ar = ''
    for k, v in vars(args).items():
        ar += f'{str(k)}: {str(v)}, '
    #ar += f'valid acc: {str(round(valid_acc, 5))}, test acc: {str(round(test_acc, 5))}'
    ar += f'test acc: {str(round(test_acc, 5))}'
    f.write(ar + '\n')
    f.write('**************\n\n\n')
    f.close()

    if args.show_result:
        print(ar)


mean = np.mean(ACC)
std = np.std(ACC)
lastprint = ' ##################################################################### \n'
lastprint += 'Mean test ACC ' + str(mean) + ',    std ' + str(std) + '\n'
lastprint += ' ##################################################################### \n'
print(lastprint)

f = open(f'{main_folder}/{namefile}.txt', 'a')
f.write(lastprint)
f.close()