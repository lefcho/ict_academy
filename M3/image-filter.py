import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def read_image(file_path):
    """Read an image from a f i l e . """
    return np.asarray(Image.open(file_path))


def display_image(img, title=''):
    plt.imshow(img, cmap=plt.get_cmap("gray"))
    plt.title(title)
    plt.axis('off')
    plt.show()


def grey_img(img):
    red = img[: ,: ,0]
    green = img[: ,: ,1]
    blue = img[: ,: ,2]

    gray_img = (0.299 * red + 0.587 * green + 0.114 * blue)
    return gray_img


def blur(image, radius=2):
    sizeX, sizeY, _ = image.shape

    px2 = np.copy(image)
    # window_area = (2 * radius + 1) ** 2

    for i in range(radius, sizeX - radius):
        for j in range(radius, sizeY - radius):
            
            # r = g = b = 0
            # for k in range(-pixels, pixels + 1):
            #     for l in range(-pixels, pixels + 1):
            #         pix = image[i + k, j + l]

            #         r += int(pix[0])
            #         g += int(pix[1])
            #         b += int(pix[2])

            block = image[i - radius : i + radius + 1, 
                          j - radius : j + radius + 1]
            
            new_pix = block.mean(axis=(0,1))
            px2[i, j] = new_pix.astype(np.uint8)

    return px2


def median(image, radius=2):
    sizeX, sizeY, _ = image.shape
    px2 = np.copy(image)

    for i in range(radius, sizeX - radius):
        for j in range(radius, sizeY - radius):
            block = image[
                i - radius : i + radius + 1,
                j - radius : j + radius + 1
            ]

            # Highest value of all pixels:
            # new_pix = block.max(axis=(0,1))

            # Actual median
            new_pix = np.median(block, axis=(0,1))
            px2[i, j] = new_pix.astype(np.uint8)

    return px2


def edge(image, radius=2):
    print(image)
    image = grey_img(image)
    sizeX, sizeY = image.shape
    px2 = np.copy(image)
    print(image)

    return image

    # for i in range(radius, sizeX - radius):
    #     for j in range(radius, sizeY - radius):
    #         block = image[
    #             i - radius : i + radius + 1,
    #             j - radius : j + radius + 1
    #         ]

    #         r = b = g = 0
    #         for k in range(-radius, radius + 1):
    #             for l in range(-radius, radius + 1):
    #                 pix = image[i + k, j + l]
    #                 if k == l == 0:
    #                     r += pix[0] * 8
    #                     g += pix[1] * 8
    #                     b += pix[2] * 8
    #                 else:
    #                     r -= pix[0]
    #                     g -= pix[1]
    #                     b -= pix[2]

    #         px2[i, j, 0] = r
    #         px2[i, j, 1] = g
    #         px2[i, j, 2] = b

    # return px2


def edge_gray(image, radius=1):
    gray = grey_img(image)        # shape (H,W)
    H, W        = gray.shape
    result      = np.zeros_like(gray)

    for i in range(radius, H-radius):
        for j in range(radius, W-radius):
            val = 0.0
            # apply a simple Laplacian‑style kernel:
            for di in range(-radius, radius+1):
                for dj in range(-radius, radius+1):
                    weight = 8 if (di==0 and dj==0) else -1
                    val   += gray[i+di, j+dj] * weight
            result[i,j] = val

    # clip to image range and convert:
    return np.clip(result, 0, 255).astype(np.uint8)


img = read_image("download.jpg")
new_arr = np.arange(75).reshape(5, 5, 3)

blur_res = blur(img, 5)
medi_res = median(img, 5)
edge_res = edge(new_arr, 2)
eg = edge_gray(img, radius=1)

# display_image(img,title="original")
# display_image(blur_res, title="Blur")
# display_image(medi_res, title="Median")
# display_image(edge_res, title="Edge")
display_image(eg, title="Edges")
