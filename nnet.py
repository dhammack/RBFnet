import numpy as np
import theano
from theano import tensor as T
from numpy.random import randn as randn
from climin import util

class Neural_Net(object):
		
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
		b1,b2,w1,w2 = util.shaped_from_flat(params_vect, template)
		self.b1.set_value(b1)
		self.b2.set_value(b2)
		self.w1.set_value(w1)
		self.w2.set_value(w2)
															
	def get_params(self, template):
		flat, (b1,b2,w1,w2) = util.empty_with_views(template)
		b1 = self.b1.get_value()
		b2 = self.b2.get_value()
		w1 = self.w1.get_value()
		w2 = self.w2.get_value()

		flat = np.concatenate(
			[b1.flatten(),b2.flatten(),w1.flatten(),w2.flatten()])
		return flat
		
	def __init__(self, input, n_in, n_nodes, n_out):
		b1_init = randn(n_nodes).reshape((n_nodes,))
		b2_init = randn(n_out).reshape((n_out,))
		w1_init = randn(n_in*n_nodes).reshape((n_in,n_nodes))
		w2_init = randn(n_nodes*n_out).reshape((n_nodes,n_out))
		
		self.b1 = theano.shared(b1_init, name='b1') #bias input->hidden
		self.b2 = theano.shared(b2_init, name='b2') #bias hidden->out
		self.w1 = theano.shared(w1_init, name='w1') #input->hidden
		self.w2 = theano.shared(w2_init, name='w2') #hidden->out
		
		act = T.nnet.sigmoid(T.dot(input, self.w1) + self.b1) #activations
		out = T.nnet.sigmoid(T.dot(act, self.w2) + self.b2) #outputs
		
		self.prob = out
		self.pred = T.round(self.prob)
		self.pred_func = theano.function([input],outputs=self.pred)
		self.prob_func = theano.function([input],outputs=self.prob)
		