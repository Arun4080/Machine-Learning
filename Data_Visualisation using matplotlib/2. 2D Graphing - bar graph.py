import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

#data
x=[5,8,10]
y=[12,16,6]

x2=[6,9,12]
y2=[9,13,10]

#plot graph
plt.bar(x,y, align = 'center')
plt.bar(x2,y2,color='g', align='center')

#for horizontal bar graph
#plt.barh(x,y, align = 'center')
#plt.barh(x2,y2,color='g', align='center')

#show what we ploted
plt.show()