import numpy as np
from glob import glob
import nrrd


class myBbox(object):

  def __init__(self):

    m2='../m1_results_folder'
    m1='../m2_results_folder'
    self.results='../save_folder/'
    self.m2_=glob('{}/*.nrrd'.format(m2))
    self.m1_=glob('{}/*.nrrd'.format(m1))



  def bbox_cal(self,data):
      a=data
      a=np.round(a)    
      x0=np.sum(a,axis=2)
      xx=np.sum(x0,axis=1)
      yy=np.sum(x0,axis=0)
      resx = next(x for x, val in enumerate(list(xx)) 
                                        if val > 0)

      resxx = next(x for x, val in enumerate(list(xx)[::-1]) 
                                        if val > 0)


      resy = next(x for x, val in enumerate(list(yy)) 
                                        if val > 0)

      resyy = next(x for x, val in enumerate(list(yy)[::-1]) 
                                        if val > 0)
      z0=np.sum(a,axis=1)
      zz=np.sum(z0,axis=0)
      resz = next(x for x, val in enumerate(list(zz)) 
                                        if val > 0)

      reszz = next(x for x, val in enumerate(list(zz)[::-1]) 
                                        if val > 0)

      return resx,resxx,resy,resyy,resz,reszz

  def execute_ensembling(self):
      for i in range(len(self.m1_)):
      	m2V,h=nrrd.read(self.m2_[i])
      	m1V,h=nrrd.read(self.m1_[i])
      	box=np.zeros(shape=(m1V.shape[0],m1V.shape[1],m1V.shape[2]))
      	bbox=self.bbox_cal(m2V)
      	box[bbox[0]:512-bbox[1],bbox[2]:512-bbox[3],bbox[4]:128-bbox[5]]=1
      	filename=self.results+m1_[i][-17:-5]+'.nrrd'
      	nrrd.write(filename,box*m1V,h)

if __name__ == '__main__':
  k=myBbox()
  k.execute_ensembling()

























