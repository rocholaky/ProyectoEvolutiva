import numpy as np
import matplotlib.pyplot as plt
pred = lambda x: 0.498*np.square(1.319*np.power(x, 2.642) + 0.29*x - 1.049)-0.653*(1.319*np.power(x, 2.642) + 0.29*x - 1.049) + 0.291
real = lambda x: np.power((1-np.power(x, 3)), 2)

x = np.random.uniform(0, 5, (80,))

sol1 = 0.498*np.square(1.319*np.power(x, 2.642) + 0.29*x - 1.049)-0.653*(1.319*np.power(x, 2.642) + 0.29*x - 1.049) + 0.291
sol2 = np.power((1-np.power(x, 3)), 2)

fig, ax = plt.subplots()
ax.plot(x, sol1, 'r*',label='predicted')
ax.plot(x, sol2,  '*',label='real')
ax.legend()
ax.set_xlabel('x')
ax.set_title('resultados predichos y reales de la función (1-x^3)^2')
plt.show()
