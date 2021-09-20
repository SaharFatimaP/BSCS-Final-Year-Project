import numpy as np

# POLYNOMIAL
x = [0, 74, 150, 162, 228, 304, 380]
y = [0.55500003, 0.49100002, 0.45000002, 0.44600002, 0.46400002, 0.48100002, 0.52400002]
pol = np.poly1d(np.polyfit(x,y,5))
#print(pol)
depth_vec = [pol(x) for x in range(381)]
print(repr(depth_vec))
