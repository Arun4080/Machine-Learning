#Arun
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

#data
x=[5,8,10]
y=[12,16,6]

x2=[6,9,12]
y2=[9,13,10]

#Plotting to our canvas
plt.plot(x,y,'g',label='line 1',linewidth=5)
plt.plot(x2,y2,'r',label='line 2',linewidth=5)

#info for notation on graph
plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

#lets show labels in graph
plt.legend()

#showing what we plotted
plt.show()