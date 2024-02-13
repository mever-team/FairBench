import numpy as np

def raw_hw(utilities, delta, weights=None):
    utilities = np.array(utilities)
    #utilities = utilities/np.sum(utilities)
    if weights is None:
        weights = [1 for _ in utilities]
    umin = np.min(utilities)
    return -delta+np.sum(np.maximum(delta+umin,utilities)*weights)


def hw2(utilities, delta, weights=None):
    utilities = np.array(utilities)
    #utilities = utilities/np.sum(utilities)
    if weights is None:
        weights = [1 for _ in utilities]
    umin = np.min(utilities)
    summands = np.minimum(np.maximum(1-delta-umin,0), 1-utilities)
    result = 1
    for summand, weight in zip(summands, weights):
        assert int(weight) == weight
        for w in range(weight):
            result = np.maximum(0, result+summand-1)
    return np.maximum(-delta+1-result, 0)


def hw(utilities, delta, weights=None):
    utilities = np.array(utilities)
    #utilities = utilities/np.sum(utilities)
    if weights is None:
        weights = [1 for _ in utilities]
    umin = np.min(utilities)
    return -delta+np.minimum(1, np.sum(np.maximum(np.minimum(delta+umin, 1),utilities)*weights))

def chw(utilities, delta, weights=None):
    utilities = np.array(utilities)
    #utilities = utilities/np.sum(utilities)
    if weights is None:
        weights = [1 for _ in utilities]
    umin = np.min(utilities)
    summands = 1-np.minimum(np.maximum(1-delta-umin,0), 1-utilities)
    result = 1-delta
    for summand, weight in zip(summands, weights):
        assert int(weight) == weight
        for w in range(weight):
            result = np.maximum(0, result+summand-1)
    return result


def rawl_chw(utilities, delta, weights=None):
    utilities = np.array(utilities)
    #utilities = utilities/np.sum(utilities)
    if weights is None:
        weights = [1 for _ in utilities]
    umin = np.min(utilities)
    summands = 1-np.minimum(np.maximum(1-delta-umin,0), 1-utilities)
    result = 1-delta
    for summand, weight in zip(summands, weights):
        assert int(weight) == weight
        for w in range(weight):
            result = np.minimum(result, summand)
    return result

def shw(utilities, delta, weights=None):
    utilities = np.array(utilities)
    #utilities = utilities/np.sum(utilities)
    if weights is None:
        weights = [1 for _ in utilities]
    umin = np.min(utilities)
    summands = 1-np.maximum(1-delta-umin-utilities, 0)
    result = 1-delta
    for summand, weight in zip(summands, weights):
        assert int(weight) == weight
        for w in range(weight):
            result = np.maximum(0, result+summand-1)
    return result


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def normalize(arr):
    return arr
    #if np.max(arr) == 0:
    #    return arr
    #return np.array(arr)/np.max(arr)

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z1 = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z1[i, j] = hw2(normalize([X[i, j], Y[i, j]]), delta=0.2)

Z2 = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z2[i, j] = chw(normalize([X[i, j], Y[i, j]]), delta=0.2)

# Determine common levels for contour plots
levels = np.linspace(min(Z1.min(), Z2.min()), max(Z1.max(), Z2.max()), 20)

fig = plt.figure()

# Plot the first contour
ax1 = fig.add_subplot(121)
contour1 = ax1.contourf(X, Y, Z1, levels=levels, cmap='inferno')
ax1.set_xlabel('$u_1$')
ax1.set_ylabel('$u_2$')
ax1.set_title('$\mathcal{P}_L(HW)$')

# Plot the second contour
ax2 = fig.add_subplot(122)
contour2 = ax2.contourf(X, Y, Z2, levels=levels, cmap='inferno')
ax2.set_xlabel('$u_1$')
ax2.set_ylabel('$u_2$')
ax2.set_title('$\mathcal{P}_L(FairHW)$')

# Add a color bar for the second contour plot
cbar_ax = fig.add_axes([0.92, 0.25, 0.03, 0.6]) # These numbers [left, bottom, width, height] are adjustable
cbar = fig.colorbar(contour2, cax=cbar_ax)

# Format colorbar tick labels to two decimal places
cbar.formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.2f}')
cbar.update_ticks()
fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)

plt.show()