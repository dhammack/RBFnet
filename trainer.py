import theano
from theano import tensor as T
import climin
import itertools
import numpy as np
from sklearn.cluster import MiniBatchKMeans as kmeans
from rbf_theano_2 import RBF_Network as theano_rbfnet

#class for training of models
class Trainer(object):
	
	def __init__(self, optimizer, num_centers, batch_size, iters):
		self.optimizer = optimizer
		self.num_centers = num_centers
		self.batch_size = batch_size
		self.max_iters = iters
		
	def build_and_train(self, X, Y):

		y_onehot = self.class_to_onehot(Y)
		n_dims = y_onehot.shape[1]
		centers = self.compute_centers(X)
		
		x = T.dmatrix()
		y = T.imatrix()
		
		#bias, centers, sigmas, weights
		template = [n_dims, centers.shape,
					self.num_centers, (self.num_centers,n_dims)]

		#initialize and train RBF network
		model = theano_rbfnet(input=x, n_cents=self.num_centers,
								centers=centers, n_dims=n_dims)

		cost = model.neg_log_likelihood(y)

		g_b = T.grad(cost, model.b)
		g_c = T.grad(cost, model.c)
		g_s = T.grad(cost, model.s)
		g_w = T.grad(cost, model.w)

		g_params = T.concatenate(
					[g_b.flatten(),g_c.flatten(),g_s.flatten(),g_w.flatten()])
					
		getcost = theano.function([x,y],outputs=cost)
		getdcost = theano.function([x,y],outputs=g_params)

		def cost_fcn(params,inputs,targets):
			model.set_params(params,template)
			x = inputs
			y = targets
			return getcost(x,y)

		def cost_grad(params, inputs, targets):
			model.set_params(params,template)
			x = inputs
			y = targets
			return getdcost(x,y)

		args = climin.util.iter_minibatches([X,y_onehot],self.batch_size,[0,0])
		batch_args = itertools.repeat(([X,y_onehot],{}))
		args = ((i,{}) for i in args)
		init_params = model.get_params(template)

		opt_sgd = climin.GradientDescent(init_params, cost_fcn, cost_grad,
							steprate=0.01, momentum=0.99, args=args,
							momentum_type="nesterov")

		opt_ncg = climin.NonlinearConjugateGradient(init_params,
													cost_fcn,
													cost_grad, args=batch_args)

		opt_lbfgs = climin.Lbfgs(init_params, cost_fcn,
								cost_grad, args=batch_args)
		#choose the optimizer
		if self.optimizer=='sgd':
			optimizer = opt_sgd
		elif self.optimizer=='ncg':
			optimizer = opt_ncg
		else: optimizer = opt_lbfgs
		
		#do the actual training.
		costs = []
		for itr_info in optimizer:
			if itr_info['n_iter'] > self.max_iters: break
			costs.append(itr_info['loss'])
			
		model.set_params(init_params, template)
		return model, costs
	
	
	def compute_centers(self,X):
		#use kmeans to compute centroids
		k = self.num_centers
		kmns = kmeans(n_clusters=k,compute_labels=False,n_init=3,max_iter=100)
		kmns.fit(X)
		return kmns.cluster_centers_

	
	@staticmethod
	def class_to_onehot(y):
		#map y from a ordinal class (1,0,2,etc)
		#to a one-hot binary vector
		classes = set(y[:])
		ynew = []
		cls_ct = len(classes)
		for yi in y:
			ynew.append([1 if yi==cls else 0 for cls in classes])
		
		return np.array(ynew)
	
	@staticmethod
	def onehot_to_int(y_onehot):
		return np.argmax(y_onehot, axis=1)
	