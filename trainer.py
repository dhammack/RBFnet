import theano
from theano import tensor as T
import climin
import itertools
import numpy as np
from sklearn.cluster import MiniBatchKMeans as kmeans
from rbf_theano_2 import RBF_Network as theano_rbfnet
from nnet import Neural_Net as nnet
from dkm import DeepKernelMachine as DKM

#class for training of models
class Trainer(object):
	
	def __init__(self, optimizer, l1_size, l2_size, batch_size, iters, reg=0.0):
		self.optimizer = optimizer
		self.l1_size = l1_size #layer 1 size
		self.l2_size = l2_size #layer 2 size
		self.batch_size = batch_size
		self.max_iters = iters
		self.penalty = reg
		
	def build_and_train_rbf(self, X, Y):

		y_onehot = self.class_to_onehot(Y)
		n_dims = y_onehot.shape[1]
		centers = self.compute_centers(X)
		
		x = T.dmatrix()
		y = T.imatrix()
		
		#bias, centers, sigmas, weights
		template = [n_dims, centers.shape,
					self.l1_size, (self.l1_size,n_dims)]

		#initialize and train RBF network
		model = theano_rbfnet(input=x, n_cents=self.l1_size,
								centers=centers, n_dims=n_dims, reg=self.penalty)

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
							steprate=0.1, momentum=0.99, args=args,
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
	
	def build_and_train_dkm(self, X, Y):

		
		y_onehot = self.class_to_onehot(Y)
		n_dims = y_onehot.shape[1]
		c1_init, c2_init = self.compute_centers(X, layers=2)
		
		x = T.dmatrix()
		y = T.imatrix()
		
		#bias, c1,c2,s1,s2, weights
		template = [(n_dims,), c1_init.shape, c2_init.shape, (self.l1_size,),
					(self.l2_size,), (self.l2_size, n_dims)]

		#initialize and train RBF network
		model = DKM(input=x, centers1=c1_init, centers2=c2_init,
					n_dims=n_dims, reg=self.penalty)

		cost = model.neg_log_likelihood(y)

		g_b = T.grad(cost, model.b)
		g_c1 = T.grad(cost, model.c1)
		g_c2 = T.grad(cost, model.c2)
		g_s1 = T.grad(cost, model.s1)
		g_s2 = T.grad(cost, model.s2)
		g_w = T.grad(cost, model.w)

		g_params = T.concatenate([g_b.flatten(),g_c1.flatten(), g_c2.flatten(),
								g_s1.flatten(),g_s2.flatten(), g_w.flatten()])
					
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
							steprate=0.1, momentum=0.99, args=args,
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
	
	
	
	def build_and_train_nnet(self, X, Y):
	
		y_onehot = self.class_to_onehot(Y)
		n_in = X.shape[1]
		n_nodes = self.l1_size
		n_out = y_onehot.shape[1]
		
		x = T.dmatrix()
		y = T.imatrix()
		
		#bias1, bias2, weights1, weights2
		template = [(n_nodes,), (n_out,), (n_in,n_nodes),(n_nodes,n_out)]

		#initialize nnet
		model = nnet(input=x, n_in=n_in, n_nodes=n_nodes, n_out=n_out)
		cost = model.neg_log_likelihood(y)

		g_b1 = T.grad(cost, model.b1)
		g_b2 = T.grad(cost, model.b2)
		g_w1 = T.grad(cost, model.w1)
		g_w2 = T.grad(cost, model.w2)
		
		g_params = T.concatenate([g_b1.flatten(),g_b2.flatten(),
									g_w1.flatten(),g_w2.flatten()])
		
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
	
	
	
	def compute_centers(self, X, layers=1):
		#use kmeans to compute centroids
		k1, k2 = self.l1_size, self.l2_size
		kmns = kmeans(n_clusters=k1, compute_labels=False,
						n_init=3, max_iter=200)
		kmns.fit(X)
		if layers==1: return kmns.cluster_centers_
		#handle only two layers right now
		Xc = kmns.transform(X)
		#cluster in transformed space
		kmns2 = kmeans(n_clusters=k2, compute_labels=False,
						n_init=3, max_iter=200)
		kmns2.fit(Xc)
		return (kmns.cluster_centers_, kmns2.cluster_centers_)
	
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
	