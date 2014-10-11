##################
# Adds rough 'seasonality' info to data set, as the data is temporal
# and appears to have some sort of pattern... let the regression decide this.
################



f=open('test_times.csv', 'wb')
g = open('test.csv', 'rb')

count = 0
period = 1091440 # divides each day into 6 periods 
				 # there are 45.840.616 entries in train.csv

for row in g:
    if count == 0:
        
        f.write('p5000,'+row)
    
    for day in range(1): 
        if (count  > 0*period + day*period*7 and count  <= 1 * period + day * period*7): p = 1
        elif (count  > 1*period + day * period*7  and count  <= 2 * period + day * period*7): p = 2
        elif (count   > 2 * period + day * period*7 and count  <= 3 * period+ day * period*7): p = 3
        elif (count   > 3 * period + day * period*7 and count  <= 4 * period+ day * period*7): p = 4
        elif (count   > 4 * period + day * period*7 and count  <= 5 * period+ day * period*7): p = 5
        elif (count   > 5 * period + day * period*7 and count  <= 6 * period+ day * period*7): p = 6
    	else: p = 6 # yes, I know I could have written better code here!
    
    if (count != 0):
        
        f.write(str(p) + ","+row)
        if count % 1000000 == 0: print "%d rows done" % count
        
    count+=1
print "Finished!"        
print count, "rows total."    
f.close()
g.close()








