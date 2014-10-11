Criteo-click-challenge-Kaggle
=============================

This is the code I used to achieve 186th place out of 718 teams for my second 'real' Kaggle challenge.  For competition details see: http://www.kaggle.com/c/criteo-display-ad-challenge

I tried two main approaches, both centered around using logistic regression to predict the probability of clicking on display ads.  The first was to try using Vowpal Wabbit, and optimize the hyperparameters using cross validation.  However, toward the end of the competition, competitor TINRTGU generously shared a from scratch implementation of single-pass online logistic regression using a simple hashing trick, along with an adaptive learning rate.  I achieved my best score by running a modified version of his code, using more bits for the hash in order to reduce collisions.

Although his code is responsible for my score, I attempted to improve on the algorithm by implementing a quadratic kernel.  Time on the competition ran out before I was successful, the main obstacle being how to efficiently implement it when using online learning given that there were an enormous amount of features after hashing.  A simpler solution, for the competition at least, would have been just to brute-force create quadratic features and then apply the algorithm as is. Several other competitors did this with improved results. I tried to avoid this as I was worried about too many collisions.

One additional thing that I tried was to add features to incorporate 'seasonality' as the training and test data were temporally ordered.  One can see from the included graph of an exponential moving average of the target data that there seems to be some cyclical nature to the data. My approach gave me slightly worse results on the test data.
