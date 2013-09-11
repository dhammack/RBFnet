from numpy.random import multivariate_normal as ndist
import numpy as np
from matplotlib import pyplot as plot
from pylab import imshow, get_cmap
from trainer import Trainer

def exotic_data_gen(ppc=300):
	#generate concentric rings
	#central: gaussian at 0,0
	#ring: sample from [-pi,pi] uniformly
	#sample from [-r,r] normally
	xvals = np.random.uniform(-5,5,size=(ppc,))
	deviations = np.random.normal(0.0,0.1,size=(ppc,)).reshape((-1,1))
	yvals = ((25-xvals**2)**0.5)
	yvals2 = -yvals
	
	locs = np.array(np.ones((ppc,2)))
	locs2 = np.ones((ppc,2))
	locs[:,0] = xvals
	locs[:,1] = yvals
	locs2[:,0] = xvals
	locs2[:,1] = yvals2
	
	#move the random deviations
	locs = locs + deviations*locs
	locs2 = locs2 + deviations*locs2
	eye = ndist([0.0,0.0],[[0.1,0],[0,0.1]],(ppc))

	X = np.concatenate([locs,locs2,eye])
	y = np.concatenate([np.zeros((ppc,)),np.ones((ppc,)),np.ones((ppc,))*2])
	return X,y
	
def gaussian_data_gen(points_per_class=500):		
	#generate some data for model evaluation/building
	ppc = points_per_class
	sig = np.array([[.3,0],[0,.3]])
	#cents = [[-2,0],[2,0],[0,2],[0,-2]]
	cents= [[-1,0],[0,0],[1,0]]
	points,targets = [],[]
	for i,c in enumerate(cents):
		targets.append(np.ones((ppc,))*i)
		points.append(ndist(c,sig,(ppc)))
		
	X = np.concatenate(points,axis=0)
	y = np.concatenate(targets,axis=0)
	inds = range(X.shape[0])
	np.random.shuffle(inds)
	Xshuf = []
	yshuf = []
	for i in inds:
		Xshuf.append(X[i])
		yshuf.append(y[i])
	Xshuf = np.array(Xshuf)
	yshuf = np.array(yshuf)
	#TODO: handle the bimodal case.
	return Xshuf,yshuf
	
	
def plot_stats(X,Y,model,costs):
	#two plots, the decision fcn and points and the cost over time
	y_onehot = Trainer.class_to_onehot(Y)
	f,(p1,p2) = plot.subplots(1,2)
	p2.plot(range(len(costs)),costs)
	p2.set_title("Cost over time")
	
	#plot points/centroids/decision fcn
	cls_ct = y_onehot.shape[1]
	y_cls = Trainer.onehot_to_int(y_onehot)
	colors = get_cmap("RdYlGn")(np.linspace(0,1,cls_ct))
	
	#model_cents = model.c.get_value()
	#p1.scatter(model_cents[:,0], model_cents[:,1], c='black', s=81)
	for curclass,curcolor in zip(range(cls_ct),colors):
		inds = [i for i,yi in enumerate(y_cls) if yi==curclass]
		p1.scatter(X[inds,0], X[inds,1], c=curcolor)
		
	nx,ny = 200, 200
	x = np.linspace(X[:,0].min()-1,X[:,0].max()+1,nx)
	y = np.linspace(X[:,1].min()-1,X[:,1].max()+1,ny)
	xv,yv = np.meshgrid(x,y)
	
	Z = np.array([z for z in np.c_[xv.ravel(), yv.ravel()]])
	Zp = Trainer.onehot_to_int(np.array(model.probability(Z)))
	Zp = Zp.reshape(xv.shape)
	p1.imshow(Zp, interpolation='nearest', 
				extent=(xv.min(), xv.max(), yv.min(), yv.max()),
				origin = 'lower', cmap=get_cmap("Set1"))
	
	p1.set_title("Decision boundaries and centroids")
	f.tight_layout()
	plot.show()					
							
	

	
def print_performance(model):
	Xnew,ynew = gaussian_data_gen()
	yhat = np.array([model.predict(x)[1] for x in Xnew])
	errs= 0
	for yh,t in zip(yhat,ynew):
		errs += 1 if yh != t else 0
	print errs,'errors.'
	
def theano_perf(model):
	Xnew,ynew = gaussian_data_gen()
	# Xnew,ynew = exotic_data_gen()
	ynew_onehot = Trainer.class_to_onehot(ynew)
	yhat = np.array(model.predict(Xnew))
	yhat = Trainer.onehot_to_int(yhat)
	errs= 0
	for yh,t in zip(yhat,ynew):
		errs += 1 if yh != t else 0
	err_rate = 100*float(errs)/ynew.shape[0]
	print 'Accuracy:',100-err_rate,'Errors:',errs
	
if __name__ == '__main__':
	#generate some training data [toy example]
	X,Y = gaussian_data_gen(points_per_class=200)
	#X,Y = exotic_data_gen(ppc=300)
	
	#initialize a model trainer.
	trainer = Trainer('ncg', num_centers=20, batch_size=30, iters=30)
	#model, costs = trainer.build_and_train_rbf(X, Y)
	model, costs = trainer.build_and_train_nnet(X,Y)
	
	#convert to binary for graphing.
	plot_stats(X, Y, model, costs)
	theano_perf(model)
	
	
	print 'done.'