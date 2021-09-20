import os


path = 'C://users//ateeb//desktop//sep annotations'
for i in os.listdir(path):
	xml = open(path + '//' + i , 'r')
	xml = xml.readlines()
	address = xml[3][7:-8]
	print(address)





