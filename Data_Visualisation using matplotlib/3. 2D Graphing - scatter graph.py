import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

#data
x=[5,8,10]
y=[12,16,6]

x2=[6,9,12]
y2=[9,13,10]

#plot graph
plt.scatter(x,y)
plt.scatter(x2,y2,color='g')

#show what we ploted
plt.show()