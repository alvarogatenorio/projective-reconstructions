import numpy as np
import matplotlib.pyplot as plt
import sys

def mirror(img):
    cols = img[:,::-1,:]
    aux1 = np.hstack((cols, img, cols))
    cors = cols[::-1,:,:]
    rows = img[::-1,:,:]
    aux2 = np.hstack((cors,rows,cors))
    img = np.vstack((aux2, aux1, aux2))

if __name__ == '__main__':
    img = plt.imread(sys.argv[1])
    # Mirror
    mirror(img)
    # Save
    plt.imsave('mirror.jpg', img)
