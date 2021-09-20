import numpy as np





def convolve( image , kernel , padding , normalization):

  if padding == True:
    p = ( kernel.shape[0] - 1 )
    if len(image.shape) == 2:
      image = image.reshape(image.shape + (1,))

    pimage = np.zeros((image.shape[0] + p , image.shape[1] + p , image.shape[2] ))
    pimage[1:-1 , 1:-1 , :] = image
    image = pimage



  if len(image.shape) == 2:
    if normalization == True:
      s = np.sum(kernel)
      kernel = kernel / s
  
    o_w = image.shape[0]  - kernel.shape[0] + 1
    o_h = image.shape[1]  - kernel.shape[1] + 1
    output = np.array([0 for i in range(o_w * o_h)] , dtype = 'float')
    output = output.reshape((o_w , o_h))
    # calculations
    for i in range(o_w):
      for j in range(o_h):
        slice = image[i:i+kernel.shape[0] , j:j+kernel.shape[0]]
        conv = slice * kernel
        output[i,j] = np.sum(conv)
        
    return output

  else: 
    if normalization == True:
      for m in range(kernel.shape[-1]):
        s = np.sum(kernel[m])
        kernel[m] = kernel[m] / s
    
    o_w = image.shape[0]  - kernel.shape[0] + 1
    o_h = image.shape[1]  - kernel.shape[1] + 1
    o_d = 3
    output = np.array([0 for i in range(o_w * o_h * o_d)])
    output = output.reshape((o_w , o_h , o_d))
    # calculations
    for k in range(3):
      for i in range(o_w):
        for j in range(o_h):
          slice = image[i:i+3 , j:j+3 , k]
          conv = slice * kernel[k]
          output[i,j,k] = np.sum(conv)
    
    return output




kernel = np.array([1,2,1,2,0,2,1,2,1]).reshape(3,3)
d = np.array([1,2,3,1,0,3,1,2,3]).reshape(3,3)w
#d = np.load('E:\\depth_dataset_backup\\1200.npy')
f = convolve(d , kernel , False , normalization = True)
print(f)







