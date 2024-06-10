import numpy as np


# sigmoid activation
def identity(input):
    return input

# sigmoid activation
def sigmoid(input):
    return 1/(1 + np.exp(-input))


class ESN:
    def __init__(self, n_inputs, n_outputs,
                 nonlinearity = 'tanh',
                 n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
                 input_scaling=None, feedback=True,
                 states_pow2=True,
                 teacher_scaling=None, teacher_shift=None,
                 leak_rate=1.0,
                 bias = 'null',
                 Euler=False, eu_step=1., diffus=1., rec_scal=1., # implement EuSN
                 random_state=None, reservoir_uniform=True, transient=20):
        """
        Echo state network with leaky-integrator neurons.
        The network can operate both in generative and predictive mode.
        :param: n_inputs: nr of input dimensions
        :param: n_outputs: nr of output dimensions
        :param: nonlinearity: this is a string describing the nonlinear activation function for the reservoir units.
                    It can be:
                    'tanh'
                    'sigmoid'
                    'identity'
        :param: n_reservoir: nr of reservoir neurons
        :param: spectral_radius: spectral radius of the recurrent weight matrix
        :param: sparsity: proportion of recurrent weights set to zero
        :param: noise: noise added to each neuron during training
        :param: input_shift: scalar or vector of length n_inputs to add to each
                    input dimension before feeding it to the network.
        :param: input_scaling: scalar or vector of length n_inputs to multiply
                    with each input dimension before feeding it to the network.
        :param: feedback: if True, feed the output/target back into output units
        :param: states_pow2: if True then it uses the power of two of the states for training and prediction.
        :param: teacher_scaling: factor applied to the target signal, when teacher_forcing=True is optim().
                    Otherwise, when teacher_forcing=False is optim() then
                    the model is trained via the feedback of the output. 
                    Therefore, in that case it is a factor applied to the output feedback signal.
        :param: teacher_shift: additive term applied to the target signal
        :param: leak_rate: parameter of leaky-integrator neurons
        :param: bias: a string that can be 'null' or 'random'.
                    if 'null' then there is no bias,
                    if 'random' then there is a random vector b 
                        uniformly distributed in (-input_scaling,input_scaling) inside the nonlinearity.
        :param: random_state: positive integer seed, np.rand.RandomState object,
                      or None to use numpy's built-in RandomState
        :param: reservoir_uniform: if True, use uniform random numbers, other use the standard normal distribution
        :param: transient: number of initial states to be discarded
        """

        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = self._correct_dimensions(input_shift, n_inputs)
        self.input_scaling = self._correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.Euler = Euler
        if Euler:
            self.eu_step = eu_step
            self.diffus = diffus
            self.rec_scal = rec_scal

        self.leak_rate = leak_rate

        self.ortho = False  # by default. If you want to use the OLESN architecture call the OLESN() method

        self.random_state = random_state
        self.reservoir_uniform = reservoir_uniform
        self.transient = transient

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.feedback = feedback

        self.states_pow2 = states_pow2

        if nonlinearity == 'tanh':
            self.actF = np.tanh
        elif nonlinearity == 'sigmoid':
            self.actF = sigmoid
        elif nonlinearity == 'identity':
            self.actF = identity
        else:
            raise Exception('It is not clear which activation function shall I use : ', nonlinearity)

        if bias == 'null':
            self.bias = np.zeros(self.n_reservoir)
        elif bias == 'random':
            # uniformly distributed bias vector in (-1,1)
            self.bias = (2 * np.random.rand(self.n_reservoir) - 1) * self.input_scaling
        else:
            raise Exception('It is not clear which bias vector shall I use : ', bias)

        # init parameters and internal variables
        self._init_vars()

    def _init_vars(self):
        """
        Initialize all relevant variables
        """

        if self.Euler:
            W = self.rec_scal * (2 * np.random.rand(self.n_reservoir, self.n_reservoir) - 1)
            W = W - W.T
            self.W_res = W - self.diffus * np.eye(self.n_reservoir)
        else:
            # initialize reservoir
            if self.reservoir_uniform is True:
                # weights uniformly distributed in [-1, 1]
                W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) * 2 - 1
                # delete a fraction of connections
                W[self.random_state_.rand(*W.shape) < self.sparsity] = 0.0
                # scale the spectral radius of the reservoir
                radius = np.max(np.abs(np.linalg.eigvals(W)))
                self.W_res = W * (self.spectral_radius / radius)
            else:
                # random normally distributed matrix with mean=0 and std=[sqrt((1.0 - self.sparsity) * self.n_reservoir)]^-1
                from math import sqrt
                W = self.random_state_.randn(self.n_reservoir, self.n_reservoir)
                # delete a fraction of connections
                W[self.random_state_.rand(*W.shape) < self.sparsity] = 0.0
                # scale the random matrix
                scale = float(self.spectral_radius) / sqrt((1.0 - self.sparsity) * self.n_reservoir)
                self.W_res = W * scale

        # random input weights uniformly distributed in [-1, 1]
        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        # random feedback (teacher forcing) weights in [-1, 1]
        self.W_fb = self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1

        # read-out weights
        self.W_out = None
        # internal states
        self.states = None

        # after-training "effective" reservoir matrix
        self.M = None

    def OLESN(self, alphas, betas, Orthog ):
        """
        This code allows to use the OLESN architecture.
        :param alphas: vector of time scales of each neuron
        :param Orthog: orthogonal matrix
        """
        self.ortho = True
        self.A = np.diag(alphas)  
        self.B = np.diag(betas)  
        self.O = Orthog  # matrix orthogonal gates

    def _correct_dimensions(self, s, target_length):
        """
        Checks the dimensionality of some numeric argument s, broadcasts it
           to the specified length if possible.

        :param s: None, scalar or 1D array
        :param target_length: expected length of s
        :return: None if s is None, else numpy vector of length target_length
        """
        if s is not None:
            s = np.array(s)
            if s.ndim == 0:
                s = np.array([s] * target_length)
            elif s.ndim == 1:
                if not len(s) == target_length:
                    raise ValueError("arg must have length " + str(target_length))
            else:
                raise ValueError("Invalid argument")
        return s

    def _update(self, state, input, output):
        """
        Performs one update step. i.e., computes the next network state by applying the recurrent weights
        to the last state and feeding in the current input and previous output

        :param state: current network state
        :param input: next input
        :param output: current output
        """

        # state update
        pre_activation = np.dot(self.W_res, state) + np.dot(self.W_in, input)

        if self.feedback:
            pre_activation += np.dot(self.W_fb, output)

        noise_term = self.noise * self.random_state_.randn(self.n_reservoir)

        if self.Euler:
            return state + self.eu_step * self.actF(pre_activation + self.bias + noise_term )
        else:
            if self.ortho:
                leak_matrix = np.dot( self.A, self.O )
                leak_term = np.dot(leak_matrix , state )
                return leak_term + np.dot( self.B , self.actF(pre_activation + self.bias + noise_term) )
            else:
                leak_term = (1.0 - self.leak_rate) * state
                return leak_term + self.leak_rate * self.actF(pre_activation + self.bias + noise_term )

    def _scale_inputs(self, inputs):
        """
        For each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument.
        :param inputs: input signal as a vector
        """

        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift

        return inputs

    def _scale_teacher(self, teacher):
        """
        Multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it.
        :param: teacher signal used for training
        :return: scaled and shifted teacher signal
        """

        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift

        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """
        Inverse operation of the _scale_teacher method.
        :param: scaled teacher signal
        :return: the original un-scaled teacher signal
        """

        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling

        return teacher_scaled

    def optim(self, opt, teacher_forcing = True,
              learning_rate = 0.001, tolerance=1e-7, batchsize=1, lambd=1, learn_every=1, regularization = 0.):
        """
        This function sets the environment for the given training method.
        :param opt: a string that defines the training method. It must be 'ridge_regr' or 'lms' or 'rls'.
        :param teacher_forcing: if True training occurs feeding back teacher signal (instead of its output).
                It facilitates learning because we don't feed back errors that might get amplified in the net.
                Nevertheless, instability issues are notorious in the transition from open-loop to closed-loop.
                Note that when opt is 'ridge_regr' then it is irrelevant the value of teacher_forcing, since it is
                necessarily used the teacher signal for it.
        :param learning_rate: learning rate of the LMS learning. If training is full-batch then it is useless.
        :param tolerance: a threshold in online algorithms for the norm of the readout corrections to halt learning.
        If the corrections are below the value tolerance then the online learning stops.
        :param batchsize: for LMS. It defines how many time steps to observe for a readout update.
        :param lambd: regularisation parameter for RLS. It is used to initialise the running estimate of the  P matrix.
               Sussillo claims that values between 1 and 100 are effective, and in any case it should be << n_reservoir.
        :param learn_every: parameter for RLS. Each learn_every time steps are updated P and the readout weights W_out.
                When learn_every=1 (default) weight updates occur at each time step of the network dynamics.
                Larger values of learn_every lighten the computational time, but they might affect learning.
        :param regularization: regularisation paramater for Ridge Regression.
        """
        self.teacher_forcing = teacher_forcing
        if opt == 'ridge_regr':
            self.regularization = regularization
        elif opt == 'lms':
            self.learning_rate = learning_rate
            self.batchsize = batchsize
            self.tolerance = tolerance
        elif opt == 'rls':
            self.lambd = lambd
            self.learn_every = learn_every
            self.tolerance = tolerance
        else:
            print('Hey, it is not clear which learning algorithm you want to use...')
        self.opt = opt

    def _ridge_regression(self, inputs, outputs):
        """
        Ridge regression optimization of read-out weights
        :param inputs: Input signal
        :param outputs: Teacher signal
        :return: Optimized output weights with ridge regression and sequence of states on the training inputs
                    NB, if self.states_pow2 is True then states contains also the power of two of the states.
        """

        # generate the entire sequence of states
        if self.states_pow2 is True:
            dim = self.n_reservoir
            states = np.zeros((inputs.shape[0], 2 * dim))  # double the reservoir dimension
            for n in range(1, inputs.shape[0]):
                states[n, :dim] = self._update(states[n - 1, :dim], inputs[n, :], outputs[n - 1, :])  # feedback teacher
                states[n, dim:] = states[n, :dim] ** 2  # COLLECT THE POWER OF TWO OF THE STATES
        else:
            states = np.zeros((inputs.shape[0], self.n_reservoir))
            for n in range(1, inputs.shape[0]):
                states[n, :] = self._update(states[n - 1], inputs[n, :], outputs[n - 1, :])  # feedback teacher

        # ridge regression (discard some initial transient states)
        first = np.dot(states[self.transient:, :].T, outputs[self.transient:, :])
        second = np.linalg.pinv(
                np.dot(states[self.transient:, :].T, states[self.transient:, :]) + self.regularization * np.identity(
                    states.shape[1]))
        w_out = np.dot(first.T, second)

        return w_out, states  # NB, if self.states_pow2 is True then states contains also the power of two of the states.

    def Least_Mean_Squares(self, inputs, outputs, startWout):
        """
        LMS is an online adaptive learning algorithm.
        It trains the LINEAR read-out weights through Stochastic Gradient Descent method.
        It optimises the instantaneous mean squared error:  e_n = 0.5*(output_n - target_n)^2.
        Linear readout is  output_n = w_out * x_n  which  implies that the antigradient is  A_n = (e_n)^T * x_n
        :param inputs: Input signal
        :param outputs: Teacher/Target signal
        :param startWout: initial readout matrix
        :return: Optimized linear readout weights and sequence of states on the training inputs
        """

        # check dimensions in case of squared states feedback
        if self.states_pow2 is True and startWout.shape[1] != 2*self.n_reservoir:
            print('When states_pow2=True the initial readout matrix should have a doubled number of columns.'
                  ' Nevertheless ', startWout.shape[1], ' is different from ', 2*self.n_reservoir)
            return None

        # initialise readout weights
        w_out = startWout

        # initialise internal state
        if self.states_pow2 is True:
            states = np.zeros((inputs.shape[0], 2*self.n_reservoir))
            # random initial internal state
            states[0, :self.n_reservoir] = 0.5 * np.random.randn(self.n_reservoir) # random state from N(0, 0.5^2)
            states[0, self.n_reservoir:] = states[0, :self.n_reservoir]**2
        else:
            states = np.zeros((inputs.shape[0], self.n_reservoir))
            # random initial internal state
            states[0, :] = 0.5 * np.random.randn(self.n_reservoir)  # random state from N(0, 0.5^2)

        # random output state
        curr_output = 0.5 * np.random.randn(self.n_outputs)  # initial output as a Gaussian random sample of N(0, 0.5^2)

        n = 1
        antigrad = np.zeros(w_out.shape)
        while n < inputs.shape[0]:

            if self.teacher_forcing:
                states[n, :self.n_reservoir] = \
                    self._update(states[n - 1, :self.n_reservoir], inputs[n, :], outputs[n - 1, :])  # feedback teacher
            else:
                # ##################### feedback current output during training (does not work well) ###################
                states[n, :self.n_reservoir] = \
                    self._update(states[n - 1, :self.n_reservoir], inputs[n, :], curr_output)  # feedback current output
                # ######################################################################################################

            if self.states_pow2 is True:
                states[n, self.n_reservoir:] = states[n, :self.n_reservoir]**2  # compute the square of the states

            x = states[n, :]
            curr_output = np.dot(w_out, x)  # compute new output
            e = outputs[n, :] - curr_output  # minus error
            antigrad += np.outer( e, x )  # cumulative antigrad matrix

            if n % self.batchsize == False: # it enters here each batchsize time steps

                antigrad = antigrad/self.batchsize  # average antigradient
                corr = self.learning_rate * antigrad  # antigradient rescaled

                norma = np.linalg.norm(corr)
                if norma >= self.tolerance:
                    w_out = w_out + corr
                else:
                    n = inputs.shape[0]

                antigrad = np.zeros(w_out.shape)  # reset antigradeint for next batch

            n += 1

        return w_out, states, norma

    def Recursive_LeastSquares(self, inputs, outputs, startWout):
        """
        The RLS converges in way fewer steps than the LSM, but each step is way computationally more expensive.
        Moreover, it is more numerically unstable than LMS.

        The RLS allows to use the proper feedback of the output for training, instead of a teacher.
        This online learning algorithm is used in FORCE learning to train the readout without relying on a teacher
        signal during training. Training can be done feeding back the current output (instead of the teacher) because
        RLS converges quickly. Converging in few steps implies that we don't feed back large errors in the net during
        training, and that makes learning feasible. In this way we avoid the stability issue related to the switch from
        open-loop to closed-loop phases.

        In LMS the antigradient is  (e_n)^T * x_n  where e_n is the current error and x_n the current internal state.
        In RLS the antigradient is  (e_n)^T * delta_n  where delta_n is built-up recursively as the internal state
            trajectory evolves. Precisely, delta_n =  P_{n-1} * x_n  where P_{n-1} is an approximation of the
            inverse of the correlation matrix of the running internal state trajectory plus a regularisation term, i.e.
            P_{n-1} \approx [ x_0*(x_0)^T + x_1*(x_1)^T + ... + x_{n-1}*(x_{n-1})^T  + I/lambd  ]^1

        :param inputs: input signal
        :param outputs: target signal
        :param startWout: initial readout matrix
        :return: w_out: trained readout weights
        :return: states: a matrix collecting the internal state trajectory of training
        :return: norma: the Euclidean norm of the last correction made to the readout weights.
        """

        # check dimensions in case of squared states feedback
        if self.states_pow2 is True and startWout.shape[1] != 2*self.n_reservoir:
            print('When states_pow2=True the initial readout matrix should have a doubled number of columns.'
                  ' Nevertheless ', startWout.shape[1], ' is different from ', 2*self.n_reservoir)
            return None

        # initialise readout weights
        w_out = startWout

        # initial approximation of the inverse of the correlation matrix of the internal state trajectory
        if self.states_pow2 is True:
            P = np.identity(2*self.n_reservoir)/self.lambd
        else:
            P = np.identity(self.n_reservoir)/self.lambd

        # initialise internal state
        if self.states_pow2 is True:
            states = np.zeros((inputs.shape[0], 2*self.n_reservoir))
            # random initial internal state
            states[0, :self.n_reservoir] = 0.5 * np.random.randn(self.n_reservoir) # random state from N(0, 0.5^2)
            states[0, self.n_reservoir:] = states[0, :self.n_reservoir]**2
        else:
            states = np.zeros((inputs.shape[0], self.n_reservoir))
            # random initial internal state
            states[0, :] = 0.5 * np.random.randn(self.n_reservoir)  # random state from N(0, 0.5^2)

        # random output state
        curr_output = 0.5 * np.random.randn(self.n_outputs)  # initial output as a Gaussian random sample of N(0, 0.5^2)

        n = 1
        while n < inputs.shape[0]:
            # evolve network state one step ahead
            if self.teacher_forcing:
                states[n, :self.n_reservoir] =\
                    self._update(states[n - 1, :self.n_reservoir], inputs[n, :], outputs[n - 1, :])  # feedback teacher
            else:
                # this works well since RLS converges quickly
                states[n, :self.n_reservoir] =\
                    self._update(states[n - 1, :self.n_reservoir], inputs[n, :], curr_output)

            if self.states_pow2 is True:
                states[n, self.n_reservoir:] = states[n, :self.n_reservoir]**2  # compute the square of the states

            # compute new output
            curr_output = np.dot(w_out, states[n, :])

            # RLS update
            if n % self.learn_every == False:  # it enters only when n is multiple of learn_every
                # update inverse correlation matrix
                x = states[n, :]
                K = np.dot(P, x)
                num = np.outer(K, K.T)  # this is equiv to np.dot(P, np.dot(x, np.dot(x.T, P)))
                scale_num = 1 + np.dot(x.T, K)
                P = P - num / scale_num

                # update the error for the linear readout
                e = outputs[n, :] - curr_output  # minus error

                # update the output weights
                # ########################## RECURSIVE LOCAL FIELD ##########################
                delta = K / scale_num
                # In LMS delta is x, here it builds up accounting the current state and recursively all previous states
                #############################################################################
                antigrad = np.outer(e, delta)  # adaptive RLS antigradient

                norma = np.linalg.norm(antigrad)
                if norma >= self.tolerance:
                    w_out = w_out + antigrad
                else:
                    n = inputs.shape[0]

            if n % 1000 == False:
                print('time step:', n, '    ||antigrad||', np.linalg.norm(antigrad))

            n += 1

        norma = np.linalg.norm(antigrad)

        return w_out, states, norma

    # ####################################################################################################################
    # ####################################################################################################################
    def FORCE_internal_weights(self, inputs, outputs, startWout, toler, lambd, timeupdates=1):

        return
    # ####################################################################################################################
    # ####################################################################################################################

    def get_internal_states(self):
        """
        Return the internal states of the network
        :return: the ESN states
        """
        
        return self.states

    def get_weight_matrices(self):
        """
        Return all weight matrices. Please note that W_out = [W_res_out, W_in_out]
        :return: all weight matrices
        """
        
        return self.W_res, self.W_in, self.W_fb, self.W_out

    def fit(self, inputs, outputs, startWout):
        """
        Learn the read-out weights by means of ridge regression.

        :param inputs: array of dimensions (N_training_samples x n_inputs)
        :param outputs: array of dimension (N_training_samples x n_outputs)
        :param startWout: is the initial readout matrix to start the online learning.
        If self.opt='ridge_regr' then learning is batch, then the matrix startWout is useless.
        :return pred_train: the network's output on the training data, using the trained weights
        :return norma: the value of the norm of the last correction matrix
        """

        # transform any vectors of shape (x, ) into vectors of shape (x, 1)
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))

        # transform input and teacher signal
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        if self.opt == 'ridge_regr':
            # ridge regression (i.e. full-batch Linear Regression with Tichonov Regularisation)
            self.W_out, states = self._ridge_regression(inputs_scaled, teachers_scaled)
        elif self.opt == 'lms':
            self.W_out, states, norma = \
                self.Least_Mean_Squares(inputs_scaled, teachers_scaled, startWout)
        elif self.opt == 'rls':
            self.W_out, states, norma = \
                self.Recursive_LeastSquares(inputs_scaled, teachers_scaled, startWout)
        else:
            print('It is not clear how I should train the model... This is the content of esn.opt:', self.opt)

        # predict on training data
        pred_train = self._unscale_teacher(np.dot(states, self.W_out.T))

        # store states
        if self.states_pow2 is True:
            self.states = states[:, :self.n_reservoir]  # take only the states, excluding their power of two
        else:
            self.states = states

        # remember the last state, input, and output, w.r.t. the training session
        self.last_state = self.states[-1, :]
        self.last_input = inputs_scaled[-1, :]
        self.last_output = teachers_scaled[-1, :]

        if self.states_pow2 is False:
            # NB when states_pow2=True matrix M doesn't make sense since W_out as a number of columns which is doubled
            self.M = self.W_res + np.dot(self.W_fb, self.W_out)  # after-training "effective" reservoir

        if self.opt == 'ridge_regr':
            return pred_train
        else:
            return pred_train, norma

    def predict(self, inputs, noise_closloop=False, continuation=True, init_cond=np.zeros(500), get_state=True):
        """
        Apply learned model on test data.

        :param inputs: array of dimensions (N_test_samples x n_inputs)
        :param noise_closloop: if True, noise is maintained during the test session
        :param continuation: if True, start the network from the last training state
        :param init_cond: the initial state to start with, when continuation=False.
        :param get_state: when True it provides in output also the internal state trajectory, if False only the output.
        :return predictions on test data
        :return when get_state=True returns also the internal state trajectory.
        """

        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))

        if not noise_closloop:
            self.noise = 0  # set noise term to zero for state update during test

        n_samples = inputs.shape[0]

        if self.states_pow2 is True:
            last_state = np.zeros(2 * self.n_reservoir)
        else:
            last_state = np.zeros(self.n_reservoir)

        if continuation:
            if self.states_pow2 is True:
                last_state[:self.n_reservoir] = self.last_state
                last_state[self.n_reservoir:] = self.last_state ** 2
            else:
                last_state = self.last_state
            last_input = self.last_input
            last_output = self.last_output
        else:
            last_state = init_cond
            last_input = np.zeros(self.n_inputs)
            last_output = np.dot(self.W_out, init_cond)

        inputs = np.vstack([last_input, self._scale_inputs(inputs)])
        if self.states_pow2 is True:
            states = np.vstack([last_state, np.zeros((n_samples, 2*self.n_reservoir))])  # double the reservoir dim
        else:
            states = np.vstack([last_state, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack([last_output, np.zeros((n_samples, self.n_outputs))])

        # process test set one sample at a time
        if self.states_pow2 is True:
            for n in range(n_samples):
                # next state
                states[n + 1, :self.n_reservoir] =\
                    self._update(states[n, :self.n_reservoir], inputs[n + 1, :], outputs[n, :])
                # compute power of two of states
                states[n + 1, self.n_reservoir:] = states[n + 1, :self.n_reservoir]**2
                # predicted output
                outputs[n + 1, :] = np.dot(self.W_out, states[n + 1, :])
            # update the states with the predictions
            self.states = np.vstack((self.states, states[:, :self.n_reservoir]))
        else:
            for n in range(n_samples):
                # next state
                states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
                # predicted output
                outputs[n + 1, :] = np.dot(self.W_out, states[n + 1, :])
            # update the states with the predictions
            self.states = np.vstack((self.states, states))

        if get_state:
            return self._unscale_teacher(outputs[1:]), states
        else:
            return self._unscale_teacher(outputs[1:])
