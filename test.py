#!/usr/bin/env python
import numpy as np
import time
array=np.zeros((5,5))
np.savetxt("test.txt",array)
l=[1,2,3,4,5,6,7,8,9,10]
print l
for i in l:
	print "\r",i,
	time.sleep(2)
	#print "list:",i,"\r",

print "This will be \r",
print "overwritten"
