import numpy as np
import matplotlib.pyplot as plt


def ShowImage(b, tit = False):
  plt.figure()
  plt.imshow(b)
  plt.colorbar()
  if tit:
    plt.title(tit)
  plt.show()


def ActivationFunction(value):
  if value >= 0:
    return 1
  else:
    return 0



def Perceptron(input, output):
  for item in input:
    np.append(item , 1)
  w0 = np.array([np.random.rand()/10 for i in range(len(input[0]))])
  iter = 0
  while(True):
    iter = iter + 1
    E = 0
    for k in range(len(input)):
      net1 = np.dot(w0, input[k])
      r1 = output[k] - ActivationFunction(net1)
      w0 = w0 + r1 * input[k]
      E = E + 0.5 * (r1)*(r1)
    if E==0:
      break
    print('Your Network is ready in ',iter,' steps')
  return w0


def Evaluate(weight , ob):
  return ActivationFunction(np.dot(weight, ob))



def main():
    I = np.array([0, 1,0, 0, 1,0, 0, 1,0])
    L = np.array([1, 0, 0,1,0 , 0,1, 1, 1])
    sampleInput = np.array([I, L])
    sampleOutput = np.array([0 , 1])
    da = Perceptron(sampleInput, sampleOutput)
    testitem = [sampleInput[0] , sampleInput[1],np.array([0, 1 ,1 , 0, 1,1 , 0,1,1]) , np.array([1, 1 ,1 , 1, 1,0 , 1,1,0]) , np.array([1, 1 ,1 , 0, 1,0 , 0,1,0])]
    op = 0
    for item in testitem:
        op = op + 1
        ShowImage(np.reshape(item,[3,3]) , 'Input '+str(op))
        ShowImage(np.reshape(sampleInput[Evaluate(da, item)],[3,3]) , 'Guessed '+str(op))






main()
