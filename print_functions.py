import numpy as np
import png


def print_img_row(dim, x, name="temp", iter="0"):
    img_size = np.shape(x)[1]
    img_collection = np.empty([img_size, img_size * dim])
    for i in range(dim):
        img_collection[:, i * img_size:(i + 1) * img_size] = np.reshape(x[i, :, :, :], [img_size, img_size]) * 255

    png.from_array(img_collection.astype(np.uint8), 'L').save("pics/" + name + "_" + str(iter) + ".png")


def print_img_matrix(dim, x, name="temp", iter="0"):
    img_size = np.shape(x)[1]
    img_collection = np.empty([img_size*dim, img_size * dim])
    for idx, image in enumerate(x):
        if idx >= dim*dim:
            break
        i = idx // dim
        j = idx % dim
        img_collection[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size] = np.reshape(image, [img_size, img_size]) * 255

    png.from_array(img_collection.astype(np.uint8), 'L').save("pics/" + name + "_" + str(iter) + ".png")