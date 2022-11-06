from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def canny(image, sigma, kernel_size):
    # Read in the image
    I = mpimg.imread(image)

    # Step 2: Create a one-dimensional Gaussian Mask G

    # Half of the kernel
    size = kernel_size // 2

    x = np.arange(-size, size + 1)
    # G = np.array([[0.011, 0.13, 0.6, 1, 0.6, 0.13, 0.011]])

    # Maps values of G to the array x
    G = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp((-x ** 2) / (2 * sigma ** 2))


    # Normalizes the kernel
    G = G / sum(G)

    # Step 3: Create 1st derivative Gaussian Masks for the x and y-direction

    # Constructs derivative of the gaussian using the identity G' = (-x / sigma ** 2) * G
    G_x = (-x / sigma ** 2) * G
    G_y = G_x

    # Step 4: Convolve the image I with G along the rows to get I_x and the columns to get
    # I_y

    # Initializes I_x and I_y
    I_x = np.zeros((I.shape[0], I.shape[1]), dtype = 'f')
    I_y = np.zeros((I.shape[0], I.shape[1]), dtype = 'f')

    # Adds padding to the image
    Ix_pad = np.pad(I, ((0,0),(size,size)), 'constant')
    Iy_pad = np.pad(I, ((size,size),(0,0)), 'constant')

    x_row_num, x_col_num = Ix_pad.shape
    y_row_num, y_col_num = Iy_pad.shape

    # Convolves G to get I_x
    for i in range(x_row_num):
        for j in range(x_col_num):
            if (j - size) < 0 or (j + size) > x_col_num - 1:
                continue
            conv = np.dot(np.flip(G), Ix_pad[i][j - size : j + size + 1])
            I_x[i][j - size] = conv

    # Convolves G to get I_y
    for i in range(y_col_num):
        col = Iy_pad[:, i].flatten()
        for j in range(y_row_num):
            if (j - size) < 0 or (j + size) > y_row_num - 1:
                continue
            conv = np.dot(np.flip(G), col[j - size : j + size + 1])
            I_y[j - size][i] = conv


    # Step 5: Convolve I_x with G_x to give I'_x and I_y with G_y to give I'_y

    # Similar process of initialization as the prev step
    I_xx = np.zeros((I.shape[0], I.shape[1]), dtype = 'f')
    I_yy = np.zeros((I.shape[0], I.shape[1]), dtype = 'f')

    Ix_pad = np.pad(I_x, ((0,0),(size,size)), 'constant')
    Iy_pad = np.pad(I_y, ((size,size),(0,0)), 'constant')

    x_row_num, x_col_num = Ix_pad.shape
    y_row_num, y_col_num = Iy_pad.shape

    # Covolutions are also similar
    for i in range(x_row_num):
        for j in range(x_col_num):
            if (j - size) < 0 or (j + size) > x_col_num - 1:
                continue
            conv = np.dot(np.flip(G_x), Ix_pad[i][j - size : j + size + 1])
            I_xx[i][j - size] = conv

    for i in range(y_col_num):
        col = Iy_pad[:, i].flatten()
        for j in range(y_row_num):
            if (j - size) < 0 or (j + size) > y_row_num - 1:
                continue
            conv = np.dot(np.flip(G_y), col[j - size : j + size + 1])
            I_yy[j - size][i] = conv

    # Step 6: Compute Magnitude of Each Pixel

    # Calculates the magnitude by mapping hypot func to I_xx and I_yy
    I_mag = G = np.hypot(I_xx, I_yy)
    I_mag = I_mag / I_mag.max() * 255

    # Step 7: Implement Non-Maxima Suppression Algorithm
    I_nme = np.zeros((I.shape[0], I.shape[1]), dtype = 'f')

    # Calculates the angle of each angle
    I_theta = np.arctan2(I_yy, I_xx) * (180 / np.pi)
    # Removes negative radians
    I_theta[I_theta < 0] += 180

    # Performs the NME
    for i in range(I_nme.shape[0] - 1):
        for j in range(I_nme.shape[1] - 1):

            q = 255
            r = 255

            # 0 Degrees
            if (0 <= I_theta[i,j] < 22.5) or (157.5 <= I_theta[i,j] <= 180):
                q = I_mag[i, j+1]
                r = I_mag[i, j-1]

            # 45 Degrees
            elif (22.5 <= I_theta[i,j] < 67.5):
                q =  I_mag[i+1, j-1]
                r =  I_mag[i-1, j+1]

            # 90 Degrees
            elif (67.5 <= I_theta[i,j] < 112.5):
                q =  I_mag[i+1, j]
                r =  I_mag[i-1, j]
            # 135 Degrees
            elif (112.5 <= I_theta[i,j] < 157.5):
                q =  I_mag[i-1, j-1]
                r =  I_mag[i+1, j+1]

            if (I_mag[i,j] >= q) and (I_mag[i,j] >= r):
                I_nme[i,j] = I_mag[i,j]


    # Step 8: Apply Hysteresis

    # Apply Basic Thresholding to label weak pixel that need to be examined with hysteresis
    I_hyst = np.zeros((I.shape[0], I.shape[1]), int)

    # high and low thresholds
    high_thresh = I_nme.max() * 0.09
    low_thresh = high_thresh * 0.05

    # Define pixel values for strong and weak pixels
    strong_pix = 255
    weak_pix = 30

    # Strong pixels are any that fall above the HIGH threshold
    strong_i, strong_j = np.where(I_nme >= high_thresh)

    # Weak pixels as any that fall BETWEEN the HIGH and LOW thresholds
    weak_i, weak_j = np.where((I_nme < high_thresh) & (I_nme > low_thresh))

    # Calculate all strong pixels
    I_hyst[strong_i, strong_j] = strong_pix

    # Calculate all weak pixels
    I_hyst[weak_i, weak_j] = weak_pix

    # Perform hysteresis to assess weak pixels

    for i in range(1, I_hyst.shape[0]-1):
        for j in range(1, I_hyst.shape[1]-1):
            if (I_hyst[i,j] == weak_pix):
                # Checks to see if neighboring pixel is a strong pixel
                n = ((I_hyst[i+1][j-1] == strong_pix) or (I_hyst[i+1][j] == strong_pix) or (I_hyst[i+1][j+1] == strong_pix)
                    or (I_hyst[i][j-1] == strong_pix) or (I_hyst[i][j+1] == strong_pix)
                    or (I_hyst[i-1][j-1] == strong_pix) or (I_hyst[i-1][j] == strong_pix) or (I_hyst[i-1][j+1] == strong_pix))

                I_hyst[i][j] = strong_pix if n else 0


    return (I_x, I_y, I_xx, I_yy, I_nme, I_hyst)



#Output of three images
out_1 = canny("image_1.jpg", 1, 5)
out_2 = canny("image_2.jpg", 1, 5)
out_3 = canny("image_3.jpg", 1, 5)

# Output of first image
pic_rows, pic_cols = (2, 3)

plt.subplot(pic_rows, pic_cols, 1)
Ix_out = plt.imshow(out_1[0], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 2)
Iy_out = plt.imshow(out_1[1], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 3)
Ixx_out = plt.imshow(out_1[2], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 4)
Iyy_out = plt.imshow(out_1[3], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 5)
Iyy_out = plt.imshow(out_1[4], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 6)
Iyy_out = plt.imshow(out_1[5], cmap="gray")
plt.axis('off')

plt.show()


# Output of second image
pic_rows, pic_cols = (2, 3)

plt.subplot(pic_rows, pic_cols, 1)
Ix_out = plt.imshow(out_2[0], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 2)
Iy_out = plt.imshow(out_2[1], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 3)
Ixx_out = plt.imshow(out_2[2], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 4)
Iyy_out = plt.imshow(out_2[3], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 5)
Iyy_out = plt.imshow(out_2[4], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 6)
Iyy_out = plt.imshow(out_2[5], cmap="gray")
plt.axis('off')

plt.show()


# Output of third image
pic_rows, pic_cols = (2, 3)

plt.subplot(pic_rows, pic_cols, 1)
Ix_out = plt.imshow(out_3[0], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 2)
Iy_out = plt.imshow(out_3[1], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 3)
Ixx_out = plt.imshow(out_3[2], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 4)
Iyy_out = plt.imshow(out_3[3], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 5)
Iyy_out = plt.imshow(out_3[4], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 6)
Iyy_out = plt.imshow(out_3[5], cmap="gray")
plt.axis('off')

plt.show()

# Output of different sigmas

sig_1 = canny("image_2.jpg", 1, 5)
sig_2 = canny("image_2.jpg", 3, 5)
sig_3 = canny("image_2.jpg", 6, 5)

pic_rows, pic_cols = (1, 3)

# Out sigmas
plt.subplot(pic_rows, pic_cols, 1)
Ix_out = plt.imshow(sig_1[5], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 2)
Iy_out = plt.imshow(sig_2[5], cmap="gray")
plt.axis('off')

plt.subplot(pic_rows, pic_cols, 3)
Ixx_out = plt.imshow(sig_3[5], cmap="gray")
plt.axis('off')

plt.show()
