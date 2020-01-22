import cv2
import numpy as np


def toBinaryImage(image, threshSize):
   image = toGreyScale(image)
   image = threshf(image, threshSize)
   return image


def erosion(image, kernelSize):
   kernel = np.ones((kernelSize, kernelSize), np.uint8)
   return cv2.erode(image, kernel, iterations=1)


def toGreyScale(image):
   return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def threshf(image, threshSize):
   (thresh, image) = cv2.threshold(image, threshSize, 255, cv2.THRESH_BINARY)
   return image


def dilation(image, kernelSize):
   kernel = np.ones((kernelSize, kernelSize), np.uint8)
   return cv2.dilate(image, kernel, iterations=1)


def get_between_range(image, lower_range, upper_range):
   # Convert BGR to HSV
   # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   #
   # hsv[:,:,0]+=180
   # hsv[:, :, 0]%=360
   # cv2.imshow('hsv', hsv)

   # Threshold the HSV image to get only blue colors
   mask = cv2.inRange(image, lower_range, upper_range)

   return mask

   # Bitwise-AND mask and original image
   # res = cv2.bitwise_and(image, image, mask=mask)
   #
   # cv2.imshow('image', image)
   # cv2.imshow('mask', mask)
   # cv2.imshow('res', res)
   # cv2.waitKey()


def find_rect(image):
   # grey_image = toGreyScale(image)
   binery_image = get_between_range(image, np.array([40, 40, 230]),
                                    np.array([200, 200, 255]))  # BGR , min, max found by trial and err
   # toBinaryImage(image, 150)

   e_kernal_size = 5
   d_kernal_size = 15

   eroaded = erosion(binery_image, e_kernal_size)
   dileded = dilation(eroaded, d_kernal_size)

   rects = getRectangles(dileded)


   # cv2.imshow("original", image)
   # cv2.imshow("binary", binery_image)
   # cv2.imshow("eroaded", eroaded)
   # cv2.imshow("dileded", dileded)

   image_with_rects = image
   for rect in rects:
       # [x0, y0, x1, y1] = rect
       image_with_rects = cv2.rectangle(image_with_rects, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)

   # cv2.imshow("image with rectangles", image_with_rects)
   cv2.waitKey()
   cv2.destroyAllWindows()
   return get_bigest_area_rect(rects)


def getRectangles(image):
   contours, _ = cv2.findContours(image, 1, 2)
   # finding the largest rectangles(area)
   rect1 = [0, 0, 0, 0]
   rect2 = [0, 0, 0, 0]
   area1 = 0
   area2 = 0

   rects = []

   for cnt in contours:
       x, y, w, h = cv2.boundingRect(cnt)

       rects.append([x, y, x + w, y + h])
       # if w * h > area1:
       #     rect1 = [x, y, x + w, y + h]
       #     area1 = w * h
       # elif w * h > area2:
       #     rect2 = [x, y, x + w, y + h]
       #     area2 = w * h

   # the rectangles are returned as [x1, y1, x2, y2]
   return rects


def get_bigest_area_rect(rects):
   max_area = 0
   chosen_rect = None

   for rect in rects:
       # [x0, y0, x1, y1] = rect
       area = (rect[2] - rect[0]) * (rect[3] - rect[1])
       if area > max_area:
           max_area = area
           chosen_rect = rect
       # image_with_rects = cv2.rectangle(image_with_rects, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)

   return chosen_rect


#TODO Median, normal avg
def get_avg_color(image):
    # return cv2.mean(image)

    avg_color_per_row = np.average(image, axis=0)
    avg_colors = np.average(avg_color_per_row, axis=0)
    return avg_colors
    #
    # # return average
    # pixels = np.float32(image.reshape(-1, 3))
    #
    # n_colors = 3
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    # flags = cv2.KMEANS_RANDOM_CENTERS
    #
    # _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    # _, counts = np.unique(labels, return_counts=True)
    #
    #
    # dominant = palette[np.argmax(counts)]
    # return dominant

def image_in_color(color):
    image = np.zeros((300, 300, 3), np.uint8)
    image[:] = color
    #find_rect(image
    return image

def get_colors_offset(color,offset):
    # print(color)
    b,g,r = color[0], color[1],color[2]
    b_max, g_max,r_max = min(255, b+offset), min(255, g+offset),min(255, r+offset)
    b_min, g_min,r_min = max(0, b-offset), max(0, g-offset),max(0, r-offset)

    min_color = (b_min, g_min,r_min)
    max_color = (b_max, g_max,r_max)
    return min_color, max_color


def slpit_to_tiles(im, x_parts, y_parts):
    image = im.copy()
    imgheight = image.shape[0]
    imgwidth = image.shape[1]

    y1 = 0
    M = imgheight//y_parts
    N = imgwidth//x_parts

    ret = [[[] for i in range(x_parts)] for j in range(y_parts)]
    # print(ret)
    for y in range(0,imgheight-1,M):
        for x in range(0, imgwidth, N):
            y1 = y + M
            x1 = x + N
            tiles = image[y:y+M,x:x+N]

            # cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0))
            # print(y//M,x//N)
            ret[y//M][x//N] = (tiles)

    # cv2.imshow("divided to rects", image)

    to_nparr = np.array(ret)

    # resaped = to_nparr.reshape(M,N)
    return ret
    # cv2.imwrite("asas.png",im)

def remove_background(image,offset):
    avg = get_avg_color(image)
    cv2.imshow("avg_color", image_in_color(avg))
    min_range, max_range = get_colors_offset(avg, offset)

    background_image = get_between_range(image, min_range, max_range)

    without_background = cv2.bitwise_not(background_image)
    return without_background


def gradiant_finder(image):
    laplacian64 = cv2.Laplacian(image, cv2.CV_64F)
    sobelx64 = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely64 = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    laplacian = np.uint8(np.absolute(laplacian64))
    sobelx = np.uint8(np.absolute(sobelx64))
    sobely = np.uint8(np.absolute(sobely64))

    # cv2.imshow("laplacian",laplacian)
    # cv2.imshow("sobelx",sobelx)
    # cv2.imshow("sobely",sobely)
    return laplacian

def combine_images(images):
    vis = []
    for x in range(len(images)):
        vis_new = images[x][0]
        for y in range(1,len(images[x])):
            vis_new = np.concatenate((vis_new, images[x][y]), axis=1) #horizontly

        if(len(vis) == 0): vis = vis_new
        else: vis = np.concatenate((vis, vis_new), axis=0)
    return vis

def main():
    path = r'C:\Users\t8814951\OneDrive\Year1_SemesterElf\Project refeal\image_procssing_simple\Images\drone_image_4.jpg'


    image = cv2.imread(path)
    cv2.imshow("original", image)

    thereshold = 80

    blocks = slpit_to_tiles(image, 25,25)
    backgrounds = [[[] for i in range(len(blocks[0]))] for j in range(len(blocks))]

    for x in range(len(blocks)):
        for y in range(len(blocks[x])):
            backgrounds[x][y] = remove_background(blocks[x][y],thereshold)
            # cv2.imshow(str(x)+", "+str(y), backgrounds[x][y])

    combined_image = combine_images(backgrounds)
    cv2.imshow("combined", combined_image)

    image_without_drone = remove_background(image,thereshold)
    # cv2.imshow("image_without_drone",image_without_drone)

    # kernal_size = 2
    eroaded = erosion(combined_image,2)
    dileded = dilation(eroaded,5)

    # cv2.imshow("eroaded",eroaded)
    cv2.imshow("dileded",dileded)

    grey = toGreyScale(image)
    cv2.imshow("grey", grey)
    laplas = gradiant_finder(grey)
    laplas = threshf(laplas, 50)
    cv2.imshow("laplas", laplas)

    # eroaded = erosion(combined_image, 2)
    dileded = dilation(laplas, 5)
    cv2.imshow("dileded", dileded)


    cv2.waitKey()


if __name__ == '__main__':
    main()


#TODO anomly detection
#np.gradiant