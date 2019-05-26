import numpy as np
import matplotlib.pyplot as plt
import sys

# Transforms a color image to a grayscale one
def to_grayscale(img):
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# Computes the integral image of a given image
def integral(img):
    shape = img.shape
    integ = np.array([img[0, 0]])
    integ = integ[np.newaxis, :]
    for j in range(1, shape[1]):
        integ = np.append(integ, [[img[0, j] + integ[0, j - 1]]], axis = 1)
    for i in range(1, shape[0]):
        acum = 0
        aux = np.array([])
        aux = aux[np.newaxis, :]
        for j in range(0, shape[1]):
            acum = acum + img[i, j]
            aux = np.append(aux, [[acum + integ[i - 1, j]]], axis = 1)
        integ = np.append(integ, aux, axis = 0)
    return integ

if __name__ == '__main__':
    # Reading image
    img = plt.imread(sys.argv[1])
    # Computing its grayscale counterpart
    img = to_grayscale(img)
    # Saving the integral image
    plt.imsave('integral.jpg', integral(img), cmap = 'gray')
