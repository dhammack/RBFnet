import numpy as np
import theano
from theano import tensor as T
from numpy.random import randn as randn
from climin import util

class RBF_Network(object):
		
	def neg_log_likelihood(self, y):
		return T.mean(T.nnet.binary_crossentropy(self.prob,y))
	
	def errors(self, y):
		return T.mean(T.neq(self.pred,y))
		
	def predict(self, X):
		return self.pred_func(X)
		
	def probability(self, X):
		return self.prob_func(X)
		
	def set_params(self, params_vect, template):
		#sets the parameters from a vector (used in climin)
		b,c,s,w = util.shaped_from_flat(params_vect,template)
		self.b.set_value(b)
		self.c.set_value(c)
		self.s.set_value(s)
		self.w.set_value(w)
															
	def get_params(self, template):
		flat, (b,c,s,w) = util.empty_with_views(template)
		b = self.b.get_value()
		c = self.c.get_value()
		s = self.s.get_value()
		w = self.w.get_value()

		flat = np.concatenate(
			[b.flatten(),c.flatten(),s.flatten(),w.flatten()])
		return flat
		
	def __init__(self,input,n_cents,centers,n_dims):
		bias_init = randn(n_dims)
		cents_init = centers
		sigmas_init = np.abs(randn(n_cents))
		weights_init = randn(n_cents*n_dims).reshape((n_cents,n_dims))
		
		self.b = theano.shared(bias_init, name='b') #bias
		self.c = theano.shared(cents_init, name='c')
		self.s = theano.shared(sigmas_init, name='s')
		self.w = theano.shared(weights_init, name='w')
		
		#thanks to comments by Pascal on the theano-users group,
		#the idea is to use 3d tensors
		C = self.c[np.newaxis, :, :]
		X = input[:, np.newaxis, :]
		
		difnorm = T.sum((C-X)**2, axis=-1).T
		
		a = T.exp(-difnorm.T * self.s.T).T
		
		self.prob = T.nnet.sigmoid(T.dot(a.T, self.w) + self.b)
		self.pred = T.round(self.prob)
		self.pred_func = theano.function([input],outputs=self.pred)
		self.prob_func = theano.function([input],outputs=self.prob)
		