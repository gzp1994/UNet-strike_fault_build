import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import load_model
#from unetd import mybce
import matplotlib.pyplot as plt
from scipy.interpolate import interpn


def main(argv):
  loadModel(argv[0])
  goFakeValidation()

def loadModel(mk):
  global model
  model = load_model('./check/unetadam.25.hdf5')

def goFakeValidation():
  n1, n2, n3 = 256, 256, 256
  seisPath  = "./train/data2/"
  predPath  = "./train/label2/"
  #ks = [100,101,102,103,106,108,110,112,113,114,115,118,119]
  # ks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
  ks = [101]
  for k in ks:
    fname = str(k)
    gx = loadData(n1,n2,n3,seisPath,fname+'.dat')
    gs = np.reshape(gx,(1,n1,n2,n3,1))
    fp = model.predict(gs,verbose=1)
    fp = fp[0,:,:,:,0]
    ft = np.transpose(fp)
    ft.tofile(predPath+fname+".dat",format="%4")
    os.popen('./goDisplay valid '+fname).read()



def loadData(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  gm,gs = np.mean(gx),np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.reshape(gx,(n3,n2,n1))
  gx = np.transpose(gx)
  return gx

if __name__ == '__main__':
    main(sys.argv)


