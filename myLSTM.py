'''
My implement of LSTM base a RNN model writen by trask.
A blog describe the RNN: {http://blog.csdn.net/zzukun/article/details/49968129}
I just modify this RNN achieve to LSTM.
'''

import numpy as np
import copy

# our sigmoid function
def sigmoid(x):
	return 1/(1+np.exp(-x))

# get sigmoid func's derivation
def deri_sigm(x):
	return x*(1-x)

# our tanh function
def tanh(x):
	return np.tanh(x)

# tanh's derivation
def deri_tanh(x):
	return 1 - x**2

# reduce middle useless dimension
def reduceDim(m):
	out = []
	for e in m:
		out.append(e[0])
	return np.array(out)

#calculate thr sumouter result of [8*12] and [8*15] is [12 * 15]
def sumouter(us,vs,lo=-1.0,hi=1.0,out=None):
    """Sum the outer products of the `us` and `vs`.
    Values are clipped into the range `[lo,hi]`.
    This is mainly used for computing weight updates
    in logistic regression layers."""
    result = out or np.zeros((len(us[0]),len(vs[0])))
    for u,v in zip(us,vs):
        result += np.outer(np.clip(u,lo,hi),v)
    return result

#set random seed
np.random.seed(0)

int2bin = {}
bin_dim = 8

largest_num = pow(2 , bin_dim)

binary = np.unpackbits(np.array([range(largest_num)],dtype=np.uint8).T , axis=1)

for i in range(largest_num):
	int2bin[i] = binary[i]

alpha = 0.1
l0_dim = 2
l1_dim = 24
l2_dim = 1

#initialize synapsis weights
syn_0 = 2 * np.random.random((l0_dim , l1_dim)) - 1
syn_1 = 2 * np.random.random((l1_dim , l2_dim)) - 1
syn_h = 2 * np.random.random((l1_dim , l1_dim)) - 1

#initialize lstm weights
WGI = 2 * np.random.random((l1_dim,1+l0_dim+l1_dim)) - 1
WGF = 2 * np.random.random((l1_dim,1+l0_dim+l1_dim)) - 1
WGO = 2 * np.random.random((l1_dim,1+l0_dim+l1_dim)) - 1
WCI = 2 * np.random.random((l1_dim,1+l0_dim+l1_dim)) - 1

syn_0_update = np.zeros_like(syn_0)
syn_1_update = np.zeros_like(syn_1)
syn_h_update = np.zeros_like(syn_h)

for i in range(40000):

	overAllError = 0

	a_int = np.random.randint(largest_num/2)
	a = int2bin[a_int]

	b_int = np.random.randint(largest_num/2)
	b = int2bin[b_int]

	#true answer
	c_int = a_int + b_int
	c = int2bin[c_int]

	#print a,b,c

	#store our predict value
	d = np.zeros_like(c)

	l_1_values = []
	l_1_values.append(np.zeros(l1_dim))
	
	states = []
	states.append(np.zeros(l1_dim))

	gix = [[None]]*bin_dim
	gfx = [[None]]*bin_dim
	gox = [[None]]*bin_dim
	cix = [[None]]*bin_dim

	gi = [[None]]*bin_dim
	gf = [[None]]*bin_dim
	go = [[None]]*bin_dim
	ci = [[None]]*bin_dim

	gierr = [[None]]*bin_dim
	gferr = [[None]]*bin_dim
	goerr = [[None]]*bin_dim
	cierr = [[None]]*bin_dim

	stateerr = [[None]]*bin_dim

	l_2_deltas = []

	source = np.ones((bin_dim,1+l0_dim+l1_dim))

	#forward propagation
	for pos in range(bin_dim):
		X = np.array([[a[-pos-1],b[-pos-1]]])
		Y = np.array([[c[-pos-1]]])
		#print X,Y

		#---basic-RNN----------
		#l_1 = sigmoid(np.dot(X   , syn_0) + np.dot(l_1_values[-1],syn_h))
		#l_2 = sigmoid(np.dot(l_1 , syn_1))

		#---LSTM---------------
		h_t_prev = copy.deepcopy(l_1_values[-1])

		#h_t_prev.shape = (l1_dim,1)

		source[pos,0] = 1
		source[pos,1:1+l0_dim] = X
		source[pos,1+l0_dim:] = h_t_prev

		
		gix[pos] = np.dot(WGI,source[pos])
		gfx[pos] = np.dot(WGF,source[pos])
		gox[pos] = np.dot(WGO,source[pos])
		cix[pos] = np.dot(WCI,source[pos])

		gi[pos] = sigmoid(gix[pos])
		gf[pos] = sigmoid(gfx[pos])
		ci[pos] = tanh(cix[pos])


		state = ci[pos] * gi[pos] + gf[pos] * states[-1]

		states.append(copy.deepcopy(state))

		go[pos] = sigmoid(gox[pos])
		
		l_1 = tanh(state) * go[pos]

		#print l_1

		l_2 = sigmoid(np.dot(l_1 , syn_1))
		#----------------------

		l_2_error = Y - l_2
		overAllError += np.abs(l_2_error[0])

		l_2_delta = l_2_error * deri_sigm(l_2)
		l_2_deltas.append(l_2_delta)

		d[-pos-1] = np.round(l_2[0])

		l_1_values.append(copy.deepcopy(l_1))

	#backward propagation
	future_src_delta = np.zeros(l1_dim)
	for pos in reversed(range(bin_dim)):
		
		#X = np.array([[ a[pos],b[pos] ]])
		
		l_1 = l_1_values[pos+1]

		#prev_l_1 = l_1_values[pos]

		l_2_delta = l_2_deltas[pos]
		l_1_error = np.dot(l_2_delta , syn_1.T)

		#----lstm-----------
		outerr = future_src_delta + l_1_error

		goerr[pos] = deri_sigm(go[pos]) * tanh(states[pos]) * outerr
		stateerr[pos] = deri_tanh(states[pos]) * go[pos] * outerr

		if pos < bin_dim-1:
			stateerr[pos] += stateerr[pos+1] * gf[pos+1]

		if pos >0:
			gferr[pos] = deri_sigm(gf[pos]) * stateerr[pos] * states[pos-1]
		gierr[pos] = deri_sigm(gi[pos]) * stateerr[pos] * ci[pos]
		cierr[pos] = deri_tanh(ci[pos]) * stateerr[pos] * gi[pos]

		sourceerr = np.dot(gierr[pos] , WGI)
		if pos >0:
			sourceerr += np.dot(gferr[pos],WGF)
		sourceerr += np.dot(goerr[pos],WGO)
		sourceerr += np.dot(cierr[pos],WCI)


		#-------------------

		#l_1_delta = l_1_error * deri_sigm(l_1)

		syn_1_update += np.atleast_2d(l_1).T.dot(l_2_delta)
		#syn_h_update += np.atleast_2d(prev_l_1).T.dot(l_1_delta)
		#syn_0_update += X.T.dot(l_1_delta)

		future_src_delta = sourceerr[0][-l1_dim:]


	#syn_0 += syn_0_update * alpha
	syn_1 += syn_1_update * alpha
	#syn_1 += syn_1_update
	#syn_h += syn_h_update * alpha

	WGI += sumouter(reduceDim(gierr[:bin_dim])  , source[:bin_dim] )
	WGF += sumouter(reduceDim(gferr[1:bin_dim]) , source[1:bin_dim])
	WGO += sumouter(reduceDim(goerr[:bin_dim])  , source[:bin_dim])
	WCI += sumouter(reduceDim(cierr[:bin_dim])  , source[:bin_dim])

	#syn_0_update *= 0
	syn_1_update *= 0
	#syn_h_update *= 0

	if i%2000 == 0:
		print 'Error:' + str(overAllError)
		print 'Pred:' + str(d)
		print 'True:' + str(c)