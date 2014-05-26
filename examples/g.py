




import numpy as np



def pdf(x,mean,var):

    expRes  = np.exp( -1.0*((x-mean)**2)/(2*var)  )
    normDiv =  np.sqrt(var*2.0*np.pi)
    normFac = (1.0 /  normDiv)
    print "expRes ",expRes
    print "normDiv",normDiv
    print "normFac",normFac
    return normFac* expRes




print pdf(x=4.3086  ,mean= 4.3086, var=0.01 )