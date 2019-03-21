import cv2

# print("Hello")
#
#
# #SCALING
# import cv2
# import numpy as np
#
# # read the image file
# img = cv2.imread("Lenna.png")  # put the lenna.png at the same directory as the script
#
# # fx: scaling factor for width (x-axis)
# # fy: scaling factor for height (y-axis)
# res = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
#
# #OR
#
# # extract height and width of the image
# height, width = img.shape[:2]
# # resize the image
# res = cv2.resize(img, (5*width, 5*height), interpolation=cv2.INTER_CUBIC)
#
# # display the image
# cv2.imshow('rescaled', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #
#
# #SKIMAGE
# from skimage.io import imread
# from skimage.transform import resize
# from skimage.transform import rescale
#
# import matplotlib.pyplot as plt
#
# # read the image file
# img = imread("Lenna.png")
#
# # resize the image
# height, width = img.shape[:2]
# res = resize(img, (height*2, width*2))
#
# # OR
# # rescale the image by factor
# res = rescale(img, (2, 2))
#
# # display the image
# plt.figure()
# plt.imshow(res)
# plt.show()

#
# #TRANSLATION
# import cv2
# import numpy as np
#
# # read the image, 0 means loading the image as a grayscale image
# img = cv2.imread("Lenna.png", 0)
# rows,cols = img.shape
#
# # define translation matrix
# # move 100 pixels on x-axis
# # move 50 pixels on y-axis
# M = np.float32([[1, 0, -50],[0, 1, -300]])
# # translate the image
# dst = cv2.warpAffine(img, M, (cols, rows))
#
# # display the image
# cv2.imshow('img', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# from skimage.io import imread
# from skimage.transform import warp
# from skimage.transform import SimilarityTransform
#
# import matplotlib.pyplot as plt
#
# # read image
# img = imread("Lenna.png", as_grey=True)
#
# # translate the image
# tform = SimilarityTransform(translation=(-100, -50))
# warped = warp(img, tform)
#
# # display the image
# plt.figure()
# plt.imshow(warped, cmap="gray")
# plt.show()



# from skimage.io import imread
# from skimage.transform import rotate
#
# import matplotlib.pyplot as plt
#
# # read image
# img = imread("Lenna.png", as_grey=True)
#
# # rotate the image for 45 degree
# dst = rotate(img, 45)
#
# # display the rotated image
# plt.figure()
# plt.imshow(dst, cmap="gray")
# plt.show()



# #AFFINE TRANSITION
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # read the image
# img = cv2.imread("Lenna.png", 0)
# rows, cols = img.shape
#
# # select three points
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
#
# # get transformation matrix
# M = cv2.getAffineTransform(pts1, pts2)
#
# # apply Affine transformation
# dst = cv2.warpAffine(img, M, (cols, rows))
#
# # display the output
# plt.subplot(121), plt.imshow(img, cmap="gray"), plt.title('Input')
# plt.subplot(122), plt.imshow(dst, cmap="gray"), plt.title('Output')
# plt.show()

#IMAGE THREASHOLDING
#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # load the image
# img = cv2.imread("Lenna.png", 0)
#
# # apply global thresholding
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# # apply mean thresholding
# # the function calculates the mean of a 11x11 neighborhood area for each pixel
# # and subtract 2 from the mean
# th2 = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#     cv2.THRESH_BINARY, 11, 2)
#
# # apply Gaussian thresholding
# # the function calculates a weights sum by using a 11x11 Gaussian window
# # and subtract 2 from the weighted sum.
# th3 = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY, 11, 2)
#
# # display the processed images
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
#
#
# for i in range(4):
#     plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

#
# from skimage.io import imread
# from skimage.filters import threshold_local
#
# import matplotlib.pyplot as plt
#
# # read image, note that pixel values of the image are rescaled in the range of [0, 1]
# img = imread("Lenna.png", as_grey=True)
#
# # global thresholding
# th1 = img > 0.5
#
# # mean thresholding
# th2 = img > threshold_local(img, 11, method="mean", offset=2/255.)
#
# # gaussian thresholding
# th3 = img > threshold_local(img, 11, method="gaussian", offset=2/255.)
#
# # display results
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
#
# for i in range(4):
#     plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


#filters
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # read the image
# img = cv2.imread("Lenna.png")
#
# # prepare a 11x11 averaging filter
# kernel = np.ones((30, 30), np.float32)/900
# dst = cv2.filter2D(img, -1, kernel)
#
# # change image from BGR space to RGB space
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
#
# # display the result
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

#Edge detection
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # read image
# img = cv2.imread("Lenna.png", 0)
# # Find edge with Canny edge detection
# edges = cv2.Canny(img, 200, 250)
#
# # display results
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#
# plt.show()
import numpy as np
from skimage.io import imread
from skimage.feature import canny

import matplotlib.pyplot as plt

# read image
img = imread("Lenna.png", as_grey=True)

# find edge with Canny edge detection
edges = canny(img)

# display results
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
