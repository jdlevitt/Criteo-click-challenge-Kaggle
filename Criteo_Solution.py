########################
# Train, CV, and Submission script for Kaggle Criteo challenge
#
#   modified from the file at:
#   http://www.kaggle.com/c/criteo-display-ad-challenge/forums/t/10322/beat-the-benchmark-with-less-then-200mb-of-memory
#   under the license below       : )
#
#  Big thanks to Kaggle user TINRTGU for sharing his code!
#  I've left sections A - D mostly as given.
# 
#  The rest of the script is primarily my add-ons to run multiple epochs and cross-validate

'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''

################################
# My Modifications
#-------------------------------

# Multiple Epochs and CV
# regularization

#-------------------------------
# Polynomial kernel (quadratic) - my implementation waaaaaay too slow... removed.
# Shuffling impractically slow in my implementation
#--------------------------------
################################
################################


from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
#import shuffle      #needed for randomizing training set - too slow

# parameters #################################################################

train = 'train_90.csv'  # path to training file
train1 = train             # a name for when we do multiple passes and randomize
test = 'test_times.csv'  # path to testing file
CV = 'train10CV.csv'  # path to CV file

D = 2 ** 30   # number of weights use for learning (on my imac with 4gb RAM I think 29 is tops)
              # was able to use 31 on ec2 instance
alpha = .1    # learning rate for sgd optimization
reg_param = .01
epochs = 15

# function definitions #######################################################

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)


# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     D: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(csv_row, D):
    x = [0]  # 0 is the index of the bias term
    for key, value in csv_row.items():
        
        index = int(value + key[1:], 16) % D  # weakest hash ever ;)
        x.append(index)
    return x  # x contains indices of features that have a value of 1


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
         
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


# D. Update given model
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
#     
# OUTPUT:
#     w: updated model
#     n: updated count

def update_w(w, n, x, p, y):
    for i in x:
        # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # but for quadratic kernel, need to be careful with chain rule
        # note that in our case, if i in x then x[i] = 1
        #w[i] -= (p - y) * alpha * exp(-1. * n[i])
        w[i] = (1 - reg_param*alpha / (sqrt(n[i]) + 1.))* w[i] - (p - y) * alpha / (sqrt(n[i]) + 1.)

        n[i] += 1.

    return w, n


##################################
#File functions
################################

def test_submission(w,sub_file, test):
    with open(sub_file, 'w') as submission:
        submission.write('Id,Predicted\n')
        counter = 0
        for t, row in enumerate(DictReader(open(test))):
            Id = row['Id']
            del row['Id']
            x = get_x(row, D)
            p = get_p(x, w) 
            counter+=1
            submission.write('%s,%f\n' % (Id, p))
        
    submission.close()
    print counter, "entries in the submission file."


#########################################
#Training and testing functions
##########################################

def CV_error(w, cv_file):
    CV_loss = 0.
    count = 0
    in_file = open(cv_file)
    for t, row in enumerate(DictReader(in_file)):
        if t>=0:                                 #modified to shorten CV file
            y = 1. if row['Label'] == '1' else 0.
            del row['Label']  # can't let the model peek the answer
            del row['Id']  # we don't need the Id


            # get features and predictions
            x = get_x(row, D)

            p  = get_p(x, w)


            # for progress validation, useless for learning our model
            CV_loss += logloss(p, y)

            if t % 1000000 == 0 and t > 1:
        	    print('%s\tencountered: %d\tcurrent logloss: %f' % (
            	    datetime.now(), t, CV_loss/(t)))
            count += 1
        
    
    print "CV logloss is:", CV_loss/count  # careful not to divide by 0!
    in_file.close()
    return CV_loss/count
            	
    
def train_model(train_file, s, w, n, loss):
    in_file = open(train_file)
    counterthing = 0
    for t, row in enumerate(DictReader(in_file)):
        y = 1. if row['Label'] == '1' else 0.
        del row['Label']  # can't let the model peek the answer
        del row['Id']  # we don't need the Id


        # step 1, get the hashed features
        x = get_x(row, D)


        # step 2, get prediction
        p  = get_p(x, w)


        # for progress validation, useless for learning our model
        loss += logloss(p, y)

        if t % 1000000 == 0 and t > 1:
        	print('%s\tencountered: %d\tcurrent logloss: %f' % (
            	datetime.now(), t, loss/(s+t)))
        # step 3, update model with answer
        w, n = update_w(w, n, x, p, y)
        counterthing+=1
        #if t>= 41200000:                  #hot fix for weird splitting error
            #break
    s = s + counterthing  #### shitty counter for this implementation   
    in_file.close()
    return w, loss, n, s


##############################################################################
# training and testing #######################################################

# initialize our model
w = [0.] * D  # weights
n = [0.] * D  # number of times we've encountered a feature

s = 0         # a line counter

# start training a logistic regression model using online gradient descent
loss = 0.
bestCV = 10. # initializing best CV
bad_CV_count = 0

# looping of main #############################

for epoch in range(epochs):
	w, loss, n, s = train_model(train1, s, w, n, loss)
	print "Epoch", epoch + 1, "is complete."
	print "Starting cross validation..."
	CVloss = CV_error(w, CV)
	if CVloss < bestCV and epoch > 0:
	    bestCV = CVloss   #update lowest score
	    print "Best CV logloss so far... starting a new epoch after saving"
	    sub = 'submission_from_CV_ec2_reg.csv'
	    test_submission(w, sub, test)
	    #print 'Starting shuffle of training set...'
	    #shuffle.shuff(train, 'cache.csv')
	    #train1 = 'cache.csv'
	    bad_CV_count = 0
	else:
	    print "Trying another epoch."
	    #print 'Starting shuffle of training set...'
	    #shuffle.shuff(train, 'cache.csv')
	    #train1 = 'cache.csv'
	    if bad_CV_count >2: break
	    bad_CV_count+=1
	print '.......................................'    
    

#######################################################
# testing (build kaggle's submission file)
##################
print 'Preparing final submission...'
sub = 'submission_master_ec2_reg.csv'
test_submission(w, sub, test)
print "Process complete, test submission saved."


