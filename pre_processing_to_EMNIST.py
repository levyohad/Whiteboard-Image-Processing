import matplotlib.pyplot as plt
import cv2
import numpy as np

def crop_interesting_p33art(image):
    # Convert the image to grayscale
    image = np.array(image, dtype=np.uint8)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
 #   _, thresh = cv2.threshold(image_gray, 20, 255, cv2.THRESH_BINARY)

    # Find the contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest height
    max_cnt = max(contours, key=get_bounding_box_height)

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(max_cnt)

    # Crop the image to the bounding box and enlarge it by 10%
    x -= w // 8
    x = max(x,0)
    y -= h // 8
    y = max(y,0)
    w += w // 4
    h += h // 4
    image_cropped = image[y:y+h, x:x+w]
   # image_cropped = image[y:y+h+3, x:x+w+3]

    return image_cropped

def crop_interesting_part2(binary_image):
    # Find the contours in the image
    binary_image = np.array(binary_image, dtype=np.uint8)
    binary_image = cv2.convertTo(binary_image, cv2.CV_8UC1)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest height
    max_cnt = max(contours, key=get_bounding_box_height)

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(max_cnt)

    # Crop the image to the bounding box and enlarge it by 10%
    x -= w // 8
    x = max(x,0)
    y -= h // 8
    y = max(y,0)
    w += w // 4
    h += h // 4
    image_cropped = binary_image[y:y+h, x:x+w]
    return image_cropped

def crop_to_word(binary_image):
    binary_image = np.array(binary_image, dtype=np.uint8)
    # Find all contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding rectangles of all contours
    bounding_rects = [cv2.boundingRect(c) for c in contours]

    # Find the bounding rectangle that encloses all other rectangles
    left = min([x for (x, y, w, h) in bounding_rects])
    top = min([y for (x, y, w, h) in bounding_rects])
    right = max([x + w for (x, y, w, h) in bounding_rects])
    bottom = max([y + h for (x, y, w, h) in bounding_rects])

    # Add a 10% padding to the bounding rectangle
    padding = 0.1
    left = int(left - (right - left) * padding)
    top = int(top - (bottom - top) * padding)
    right = int(right + (right - left) * padding)
    bottom = int(bottom + (bottom - top) * padding)

    # Crop the image to the bounding rectangle
    cropped_image = binary_image[top:bottom, left:right]
    return cropped_image

def laplacian_filter(img):
    # Convert the image to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply the Laplacian filter
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    # Convert the result back to an 8-bit unsigned integer
    laplacian = np.uint8(np.absolute(laplacian))
    # Stack the grayscale result to form an RGB image
    result = np.dstack((laplacian, laplacian, laplacian))
    return result

def read_image(image_path,thresh):
    # Import the Image module from the PIL package
    from PIL import Image
    # Open an image file
    with Image.open(image_path) as image:
        # Convert the image to RGB mode
        image = image.convert("RGB")
        # Get the image width and height
        width, height = image.size

        # Create an empty matrix to store the pixel values
        pixel_matrix = [[0 for _ in range(width)] for _ in range(height)]

        # Iterate over the pixels in the image and set the corresponding value in the matrix
        for y in range(height):
            for x in range(width):
                r, g, b = image.getpixel((x, y))
                pixel_matrix[y][x] = 1 if (r + g + b) / 3 <= thresh else 0
        return pixel_matrix

# def read_image_and_laplace_filter(image_path, thresh):
#     # Read the image using OpenCV
#     image = cv2.imread(image_path)
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     cv2.imshow("gray", gray)
#     # Apply the Laplacian filter to the grayscale image
#     laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#     cv2.imshow("laplacian", laplacian)
#     # Convert the image to binary by applying a threshold
#     _, binary = cv2.threshold(laplacian, thresh, 255, cv2.THRESH_BINARY)
#     cv2.imshow("binary", binary)
#     cv2.waitKey(0)
#     return binary

def find_1(temp):
    temp_list = []
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if temp[i][j] == 1:
                temp_list.append([i,j])
    return temp_list

def find_closest_square_in_x(temp, x):
    temp_center = ((temp[0] + temp[1]) / 2, (temp[2] + temp[3]) / 2)  # Find the center of temp
    temp_x_center = temp_center[1]  # Find the x-coordinate of the center of temp
    # Find the centers and angles of all the squares in x
    x_centers = []
    angles = []
    for i in range(len(x)):
        square = x[i][1]
        x_center = ((square[0] + square[1]) / 2, (square[2] + square[3]) / 2)  # Find the center of the current square
        x_centers.append(x_center)
        # Calculate the angle between the centers of temp and the current square, in the [-pi, pi] range
        angle = np.arctan2(x_center[0] - temp_center[0], x_center[1] - temp_center[1])
        angles.append(angle)
    # Find the minimum x-distance, x-coordinate, and angle within the acceptable range
    min_x_distance = float('inf')
    min_x_center = float('inf')
    min_angle = float('inf')
    min_x_distance_index = -1
    min_x_center_index = -1
    min_angle_index = -1
    for i in range(len(x_centers)):
        #calculate the distance between the centers of temp and the current square
        x_distance = np.sqrt((x_centers[i][0] - temp_center[0]) * 2 + (x_centers[i][1] - temp_center[1]) * 2)
        # print("X Distance is : ", x_distance)
      #  print(angles[i])
    #    print(abs(abs(angles[i]) - np.pi / 2))
        if abs(angles[i])<0.25 or abs(angles[i])<=min_angle:
            #check if the x of the temp center is smaller than the temp_x_center
            if(abs(temp_center[0]-x_centers[i][0])<25):
                if x_centers[i][1] >= temp_x_center:
                    if x_distance < min_x_distance:
                            min_x_distance = x_distance
                            min_x_center = x_centers[i][0]
                            min_angle = abs(angles[i])
                            min_x_distance_index = i
                            min_x_center_index = i
                            min_angle_index = i
    # If an acceptable x-distance, x-coordinate, and angle were found, return the corresponding square in x
    if min_x_distance_index != -1 and min_x_center_index != -1 and min_angle_index != -1:
        return x[min_x_center_index]
    # If no acceptable x-distance, x-coordinate, and angle were found, return None
    else:
        return None

def find_closest_square(groups):
    # Find the closest square to the point (0, 0)
    closest_square = None
    min_distance = float('inf')
    temp_i = -1
    for i in range(len(groups)):
        square = groups[i][1]
        center = ((square[0] + square[1]) / 2, (square[2] + square[3]) / 2)  # Find the center of the current square
        distance = np.sqrt(center[0] * 2 + center[1] * 2)  # Find the distance between the center of the current square and (0, 0)
        if distance < min_distance:
            min_distance = distance
            closest_square = groups[i]
            temp_i = i
    # Find the closest square to the point (0, 0) in the x-direction
    # Return the closest square to the point (0, 0) in the x-direction
    return temp_i

def find_first_column(images):
    for image in images:
        for i in range(len(image[0])):
            column = [row[i] for row in image]
         #   print(column)
            if 1 in column:
                return i
    return -1

def get_bounding_box_height(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return h

def squarify(rect_image, x_pad, y_pad):
    a = rect_image.shape[0]
    b = rect_image.shape[1]
    if a > b:
        padding = ((0, 0), ((a-b)//2, (a-b)//2))
    else:
        padding = (((b-a)//2, (b-a)//2), (0,0))
    img = np.pad(rect_image, padding, mode='constant')
    return np.pad(img, ((y_pad,y_pad),(x_pad,x_pad)), mode='constant')
    # return binary_image_paddnig(img, x_pad, y_pad)

def binary_image_paddnig(img,x_pad=3,y_pad=7):
    ### input: binary image
    ### output: binary image with margins of x_pad on x axis and y pad on y axis

    padding_x_up = x_pad
    padding_x_down = x_pad
    padding_y_up = y_pad
    padding_y_down = y_pad

    for i in range(y_pad):
        if sum(img[i,:]) == 0:
            padding_y_up = padding_y_up-1
        if sum(img[-i-1,:]) == 0:
            padding_y_down = padding_y_down - 1

    for j in range(x_pad):
        if sum(img[:, j]) == 0:
            padding_x_up = padding_x_up-1
        if sum(img[:, -j-1]) == 0:
            padding_x_down = padding_x_down - 1

    return np.pad(img,((padding_y_up,padding_y_down),(padding_x_up,padding_x_down)), mode='constant')

def grayscale_and_resize(image, x_pad, y_pad):
    # Convert the image to grayscale
  #  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    square_image = squarify(image, x_pad, y_pad)
    image_resized = cv2.resize(square_image,(28,28))
    image_normalized = image_resized / 255

    # Add an extra dimension to the image to match the shape of the EMNIST dataset
    image_emnist = np.expand_dims(image_normalized, axis=2)

    return image_emnist

def num_to_letter(let):

    ### input: string
    ### output: string
    ### the function corrects chars classified as numbers to letters.
    if let == '6':
        return 'b'
    if let == '9':
        return 'g'
    if let == '0':
        return 'o'
    if let == '1':
        return 'i'
    if let == '2':
        return 'z'
    if let == '5':
        return 's'
    if let == '8':
        return 'b'
    if let == '7':
        return 't'
    if let == '4':
        return 'a'
    if let == '3':
        return 'e'
    return let

    if let == '6':
        let = 'b'
    elif let == '9':
        let = 'g'
    elif let == '8':
        let = 'i'
    elif let == '3':
        let = 'i'
    elif let == '0':
        let ='o'
    elif let == '5':
        let = 's'
    elif let == '2':
        let = 'z'
    else:
        let = let
    return let

def crop_binary_image(binary_image):
    rows, cols = np.where(binary_image == 1)
    top_row, bottom_row = np.min(rows), np.max(rows)
    left_col, right_col = np.min(cols), np.max(cols)

    return binary_image[top_row:bottom_row + 1, left_col:right_col + 1]

def pre_process_to_emnist(img_loc,i_dot_thresh, x_pad, y_pad):
    # img_cropped = crop_interesting_part(img_loc, i_dot_thresh)
    img_cropped = crop_binary_image(img_loc)
    try:
        if img_cropped == 0:
            return 0
    except:
        img_to_emnist = grayscale_and_resize(img_cropped, x_pad, y_pad)
        return img_to_emnist

def crop_interesting_part(image,i_dot_thresh):
        # Find the contours in the image
    image = np.array(image, dtype=np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest height
    try:
        max_cnt = max(contours, key=get_bounding_box_height)
    except:
        return image

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(max_cnt)

    # Crop the image to the bounding box and enlarge it by 10%
    x -= w // 8
    x = max(x, 0)
    y -= h // 8
    y = max(y, 0)
    w += w // 4
    h += h // 4
    image_cropped = image[y:y+h, x:x+ w]
    sumdots = sum(sum(image_cropped))
    if sumdots <= i_dot_thresh: # This part is for throwing the dot from the i (lowercase). needs to be adjusted.
        return 0
    #return np.pad(image_cropped, ((0,0),(8,8)), mode='constant')
    return image_cropped

    # Convert the image to grayscale
  #  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    #_, thresh = cv2.threshold(image_gray, 20, 255, cv2.THRESH_BINARY)

    # Find the contours in the image
    # #contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    #
    # # Find the contour with the largest height
    # max_cnt = max(contours, key=get_bounding_box_height)
    #
    # # Find the bounding box of the largest contour
    # x, y, w, h = cv2.boundingRect(max_cnt)
    #
    # # Crop the image to the bounding box
    # image_cropped = image[y:y+h+3, x:x+w+3]
    #
    # return image_cropped


# def get_bounding_box_height(cnt):
#     x, y, w, h = cv2.boundingRect(cnt)
#     return h
#
#
# def squarify(rect_image):
#     a = rect_image.shape[0]
#     b = rect_image.shape[1]
#     if a > b:
#         padding = ((0, 0), ((a-b)//2, (a-b)//2))
#     else:
#         padding = (((b-a)//2, (b-a)//2), (0, 0))
#     return np.pad(rect_image, padding, mode='constant')


# def grayscale_and_resize(image):
#     # Convert the image to grayscale
#     square_image = squarify(image)
#     #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     #square_image = squarify(image_gray)
#     image_resized = cv2.resize(square_image,(28,20))
#     image_normalized = image_resized / 255
#
#     # Add an extra dimension to the image to match the shape of the EMNIST dataset
#     image_emnist = np.expand_dims(image_normalized, axis=2)
#
#     return image_emnist


# def pre_process_to_emnist(img_loc):
#     img_cropped = crop_interesting_part(img_loc)
#     img_to_emnist = grayscale_and_resize(img_cropped)
#     return img_to_emnist
def get_frames_from_video(video_path, frame_interval_seconds=3): # just for us, when video is downloaded not live
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval between frames in number of frames
    frame_interval = int(frame_interval_seconds * cap.get(cv2.CAP_PROP_FPS))

    # Initialize a list to store the frames
    frames = []

    # Iterate through the video frame by frame
    for i in range(0, total_frames, frame_interval):
        # Set the position of the video file to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        # Read the frame from the video file
        success, frame = cap.read()

        # If the frame was successfully read, append it to the list of frames
        if success:
            frames.append(frame)

    # Release the video file
    cap.release()

    return frames

def to_binary_image(image, T):
    # Convert the image to grayscale if it's not already grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply thresholding to the grayscale image to get the binary image
    _, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    binary = 255 - binary
    cv2.imshow('binary',binary)
    cv2.waitKey(0)
    return binary

if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6, 6))
    # Iterate over the characters
    for i, char in enumerate('Blackbird'):
        # Get the current subplot
        ax = axes[i // 3, i % 3]
        img = plt.imread('{}.jpeg'.format(char))
        img_to_emnist = pre_process_to_emnist(img,i_dot_thresh, x_pad, y_pad)
        plt.savefig(r"C:\Users\ohadi\PycharmProjects\pythonProject3\TAMI\images_to_EMNIST\\\\" + char + ".png")
        ax.imshow(img_to_emnist)
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

