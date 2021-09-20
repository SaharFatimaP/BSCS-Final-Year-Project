import numpy as np
from PIL import Image
depth = np.load('C:\\users\\ateeb\\downloads\\depth_uint16_mm.npy')
for d in depth[:10]:
     #d = np.load( 'E:\\depth_dataset_backup\\'+str(i+1) + str(94) + '.npy')
     #d = np.load('C:\\users\\ateeb\\desktop\\155.npy')
     d = depth[7]
     d = d / np.mean(d)
     d = d * 255
     d = d + 20
     img = Image.fromarray(d)
     img = img.convert("RGB")
     #img.save('depth_jpg\\'+str(i)+'.jpg')
     img.show()
     break