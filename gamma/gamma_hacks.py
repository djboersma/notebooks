# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Gamma Dose Distribution Evaluation Tool

# <markdowncell>

# In this notebook I am playing around with the quantitative comparison method described in a paper written by Daniel A. Low with the same title as this notebook: Journal of Physics: Converence Series **250** (2010) 012071, doi:10.1088/1742-6596/250/1/012071. The implementation is still very crude, it would be nice to add options to also compute the angle, or to simplify it to give only a binary information about each pixel (True if gamma<=1, False if gamma>1).

# <markdowncell>

# First step: import relevant modules and define some test distributions. 

# <codecell>

import numpy as np
import pylab as plt
import time
N=256
edges=np.linspace(-1.,1.,N,False)+1./N
x,y=np.meshgrid(edges,edges);
# generate nonsense dose distribution for testing
z=np.exp(-1.5*x**2-1.9*y**2+0.27*x*y+0.13*x-0.25*y-3)
zz=z.copy()
# add slightly different levels of random noise, sabotage normalization a bit
z*=np.random.normal(1.03,0.05,(N,N));
zz*=np.random.normal(1.0,0.04,(N,N));

# <markdowncell>

# Define the functions that actually compute the gamma index.

# <codecell>

# z1,z2 maybe scalars or arrays
def reldiff2(z1,z2,dzmax):
    zdiff=z1-z2
    zsum=np.abs(z1+z2)+0.001 # avoid div by zero
    reldz2=(zdiff/zsum/dzmax)**2
    return reldz2

def gammacmp(z1,z2,dzmax,drmax):
    if z1.shape != z2.shape:
        return None
    import numpy as np
    g00=np.sqrt(reldiff2(z1,z2,dzmax))
    gmax=np.round(g00) # with "floor" we would speed up a bit more
    print("g00 min=%g, max=%g; %d nonzero gmax" % (np.min(g00),np.max(g00),np.sum(gmax>0.0)))
    nx,ny=z1.shape
    g2=np.zeros([nx,ny],dtype=float)
    for x in range(nx):
        for y in range(ny):
            igmax=int(gmax[x,y])
            if igmax<=0:
                g2[x,y]=g00[x,y]**2
            else:
                g2list=[]
                ixmin=max(x-igmax,0)
                ixmax=min(x+igmax+1,nx)
                iymin=max(y-igmax,0)
                iymax=min(y+igmax+1,ny)
                for ix in range(ixmin,ixmax):
                    for iy in range(iymin,iymax):
                        r2=((ix-x)**2+(iy-y)**2)/drmax
                        dz2=reldiff2(z1[x,y],z2[ix,iy],dzmax)
                        g2list.append(r2+dz2)
                g2[x,y]=min(g2list)
    g=np.sqrt(g2)
    return g

# <markdowncell>

# Now compute the gamma index for the test distributions, and report the run time (walltime).

# <codecell>

t1=time.time()
gg=gammacmp(z,zz,0.02,3.0)
t2=time.time()
print("calculation time: %g seconds" % (t2-t1))

# <markdowncell>

# Plot the result.
# (Does not always work correctly, sometimes I need to run this twice because the first time the figure comes out very small.)

# <codecell>

f1=plt.figure(1,figsize=[17,7])
f1.clf()
f1.add_subplot(121)
plt.pcolormesh(x,y,z-zz)
plt.colorbar()
plt.title("dose difference")
f1.add_subplot(122)
plt.pcolormesh(x,y,gg)
plt.colorbar()
plt.title("gamma")

# <codecell>


