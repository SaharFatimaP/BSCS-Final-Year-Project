from PIL import Image , ImageDraw
import numpy as np 

# FOR SANITY CHECK
x = np.load('C:\\users\\ateeb\\desktop\\localization\\X_cow_depth_data_inner.npy')
y = np.load('C:\\users\\ateeb\\desktop\\localization\\y_cow_depth_data_inner.npy')
'''y_hat = [[ 10.176361 ,  7.029873, 202.23273 , 159.25871 ],
[ 13.844778 , 11.046228, 206.43835  ,155.15399 ],
[ 11.768343 , 12.802155, 206.19368  ,157.80193 ],
[174.58788   , 5.480496, 370.16867  ,153.95145 ],
[ 13.940234   , 2.5926373, 217.81844  , 167.22186  ],
[ 25.13865   ,  1.6103624, 212.58319 ,  159.14532  ],
[157.43779   ,21.409328, 394.35483 , 177.29414 ],
[168.22209   ,16.687765, 376.8743  , 159.73547 ],
[166.33083   ,19.004232, 385.9164  , 150.37462 ],
[157.12668   ,17.283434, 374.90878 , 151.57472 ]]
'''
y_hat = [[1 , 9 , 203 , 155  ],
[1 , 8 , 204 , 154  ],
[1 , 7 , 200 , 153  ],
[211 , 5 , 394 , 159  ],
[3 , 9 , 202 , 147  ],
[5 , 6 , 203 , 146 ],
[197 , 21 , 417 , 172  ],
[187 , 17 , 398 , 167  ],
[176 , 19 , 396 , 167  ],
[175 , 19 , 399 , 168  ]]

count =0 
for i in zip(x[-10:],y_hat):
	img = Image.fromarray(i[0])
	draw = ImageDraw.Draw(img)
	x1 = int(i[1][0])
	y1 = int(i[1][1])
	x2 = int(i[1][2])
	y2 = int(i[1][3])
	draw.rectangle( ( (x1,y1) , (x2, y2) ) , outline="green")
	img.save('..//atif//'+str(count)+'.jpg')
	count += 1
	#img.show()

