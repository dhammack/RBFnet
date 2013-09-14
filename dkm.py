#deep kernel machine
#rbf layer -> rbf layer -> classifier

import numpy as np
import theano
from theano import tensor as T
from numpy.random import randn as randn
from climin import util

class DeepKernelMachine(object):
		
	def neg_log_likelihood(self, y):
		return T.mean(T.nnet.binary_crossentropy(self.prob,y)) \
			+ self.reg*abs(self.s1).mean() + self.reg*abs(self.s2).mean()
		#return T.mean(1-T.exp(-(self.prob-y)*(self.prob-y)))
		
	def errors(self, y):
		return T.mean(T.neq(self.pred,y))
		
	def predict(self, X):
		return self.pred_func(X)
		
	def probability(self, X):
		return self.prob_func(X)
		
	def set_params(self, params_vect, template):
		#sets the parameters from a vector (used in climin)
		b, c1, c2, s1, s2, w = util.shaped_from_flat(params_vect, template)
		self.b.set_value(b)
		self.c1.set_value(c1)
		self.c2.set_value(c2)
		self.s1.set_value(s1)
		self.s2.set_value(s2)
		self.w.set_value(w)
															
	def get_params(self, template):
		flat, (b, c1, c2, s1, s2, w) = util.empty_with_views(template)
		b = self.b.get_value()
		c1 = self.c1.get_value()
		c2 = self.c2.get_value()
		s1 = self.s1.get_value()
		s2 = self.s2.get_value()
		w = self.w.get_value()

		flat = np.concatenate(
			[b.flatten(),c1.flatten(), c2.flatten(), s1.flatten(), s2.flatten(), w.flatten()])
		return flat
		
	def __init__(self, input, centers1, centers2, n_dims, reg):
		n_cents1, n_cents2 = centers1.shape[0], centers2.shape[0]
		
		bias_init = randn(n_dims).reshape((n_dims,))
		#cents1_init = randn(centers1.shape[0]*centers1.shape[1]).reshape(centers1.shape)
		cents2_init = randn(centers2.shape[0]*centers2.shape[1]).reshape(centers2.shape)
		cents1_init = centers1
		#cents2_init = centers2
		
		sigmas1_init = randn(n_cents1).reshape((n_cents1,))
		sigmas2_init = randn(n_cents2).reshape((n_cents2,))
		weights_init = randn(n_cents2*n_dims).reshape((n_cents2, n_dims))
		
		#regularization
		self.reg = reg
		
		#params
		self.b = theano.shared(bias_init, name='b', borrow=True) #bias
		self.c1 = theano.shared(cents1_init, name='c1', borrow=True)
		self.c2 = theano.shared(cents2_init, name='c2', borrow=True) #2nd lvl centroids
		self.s1 = theano.shared(sigmas1_init, name='s1', borrow=True)
		self.s2 = theano.shared(sigmas2_init, name='s2', borrow=True)
		self.w = theano.shared(weights_init, name='w', borrow=True)
		
		#thanks to comments by Pascal on the theano-users group,
		#the idea is to use 3d tensors
		C1 = self.c1[np.newaxis, :, :]
		X1 = input[:, np.newaxis, :]
		difnorm = T.sum((C1-X1)**2, axis=-1)
		a1 = T.exp(-difnorm * (self.s1**2)) #activations of first layer (dims x examples)
		C2 = self.c2[np.newaxis, :, :]
		X2 = a1[:, np.newaxis, :]
		difnorm2 = T.sum((C2-X2)**2, axis=-1)
		a2 = T.exp(-difnorm2 * (self.s2**2))
		
		self.prob = T.nnet.sigmoid(T.dot(a2, self.w) + self.b)
		self.pred = T.round(self.prob)
		self.pred_func = theano.function([input],outputs=self.pred)
		self.prob_func = theano.function([input],outputs=self.prob)
		
		