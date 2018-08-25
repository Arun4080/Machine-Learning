import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("ggplot")

#exract x,y from csv file
x,y = np.loadtxt('data_files/example1.csv',unpack=True,delimiter = ',')

#plot our x,y co-ordinates
plt.plot(x,y)

plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

#show what we ploted
plt.show()

