import numpy as np
import noise 
import matplotlib.pyplot as plt

X = [float(i)/1000 for i in range(1, 2000)]
y = [0 for x in X]
s = 1.0


white = noise.white(X, y, s, 1.0)
pink = noise.pink(X, y, s, 1.0)
red = noise.red(X, y, s, 1.0)
blue = noise.blue(X, y, s, 1.0)
violet = noise.violet(X, y, s, 1.0)

#visualize white noise
plt.scatter(X, white)
plt.title("white")
plt.show()
plt.clf()

#visualize pink noise
plt.scatter(X, pink)
plt.title("pink")
plt.show()
plt.clf()

#visualize red noise
plt.scatter(X, red)
plt.title("red")
plt.show()
plt.clf()

#visualize blue noise
plt.scatter(X, blue)
plt.title("blue")
plt.show()
plt.clf()

#visualize violet noise
plt.scatter(X, violet)
plt.title("violet")
plt.show()
plt.clf()
