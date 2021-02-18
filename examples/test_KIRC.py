import sys, os
#sys.path.append("/home/travers/WindowsDesktop/cox-nnet/cox_nnet")
sys.path.append("./cox-nnet/cox-nnet/")

from cox_nnet import cox_nnet as cnn
import numpy
import sklearn
from sklearn.model_selection import train_test_split as tts

# load data
x = numpy.loadtxt(fname="./cox-nnet/examples/KIRC/log_counts.csv.gz",delimiter=",",skiprows=0)
ytime = numpy.loadtxt(fname="./cox-nnet/examples/KIRC/ytime.csv",delimiter=",",skiprows=0)
ystatus = numpy.loadtxt(fname="./cox-nnet/examples/KIRC/ystatus.csv",delimiter=",",skiprows=0)
print("1 EX2 KIRC: end of loading data")

# split into test/train sets
x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = \
    tts(x, ytime, ystatus, test_size = 0.2, random_state = 1)
    #sklearn.cross_validation.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = 1)
print("2 EX2 KIRC: end of splitting into test/train sets")

# split training into optimization and validation sets
x_opt, x_validation, ytime_opt, ytime_validation, ystatus_opt, ystatus_validation = \
    tts(x_train, ytime_train, ystatus_train, test_size = 0.2, random_state = 123)
    #sklearn.cross_validation.train_test_split(x_train, ytime_train, ystatus_train, test_size = 0.2, random_state = 123)
print("3 EX2 KIRC: end of splitting training into optimization/validation sets")

# set parameters
model_params = dict(node_map = None, input_split = None)
search_params = dict(method = "nesterov", learning_rate=0.01, momentum=0.9, 
    max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
    rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0)
cv_params = dict(cv_metric = "cindex", L2_range = numpy.arange(-3,1.67,0.33))
print("4 EX2 KIRC: end of setting parameters (3 dictionaries)")

#profile log likelihood to determine lambda parameter
likelihoods, L2_reg_params = cnn.L2Profile(x_opt,ytime_opt,ystatus_opt,
    x_validation,ytime_validation,ystatus_validation,
    model_params, search_params, cv_params, verbose=False)
print("5 EX2 KIRC: end of using profile log likelihood to determine lambda parameter")
numpy.savetxt("cox-nnet/examples/CZ/KIRC_likelihoods.csv", likelihoods, delimiter=",")
print("6 EX2 KIRC: end of saving KIRC likelihoods to .txt file named KIRC_likelihoods.csv")

#build model based on optimal lambda parameter
L2_reg = L2_reg_params[numpy.argmax(likelihoods)]
model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
model, cost_iter = cnn.trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)
print("7 EX2 KIRC: end of building and training model based on optimal lambda parameter")
theta = model.predictNewData(x_test)
print("8 EX2 KIRC: end of predicting new data using function predictNewData")

numpy.savetxt("./cox-nnet/examples/CZ/KIRC_theta.csv", theta, delimiter=",")
numpy.savetxt("./cox-nnet/examples/CZ/KIRC_ytime_test.csv", ytime_test, delimiter=",")
numpy.savetxt("./cox-nnet/examples/CZ/KIRC_ystatus_test.csv", ystatus_test, delimiter=",")

print("END OF KIRC EXAMPLE")
