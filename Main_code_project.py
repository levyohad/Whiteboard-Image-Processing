from sklearn.cluster import DBSCAN
from autocorrect import Speller
from pre_processing_to_EMNIST import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from keras import backend as K
from keras.models import model_from_json
import warnings
import cv2
import torch
import time
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
#from scratch_17 import model
import tkinter as tk
import matplotlib.patches as patches
import random
import threading


class CameraScannerThread(threading.Thread):
    def __init__(self, sample_interval):
        super().__init__()
        self.sample_interval = sample_interval
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            start_time = time.time()
            # do camera scanning here
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time < self.sample_interval:
                time.sleep(self.sample_interval - elapsed_time)

    def stop(self):
        self.stop_event.set()


class GuiThread(threading.Thread):
    def __init__(self):
        super().__init__()
       # self.gui = gui
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            # handle gui interactions here
            time.sleep(0.01)

    def stop(self):
        self.stop_event.set()

warnings.filterwarnings('ignore')

def separate_objects(img,seperate_obj_x_thresh):
    ## This function divides binary image to binary images of object based on their contur
    ## input: binary image
    ## ouput: list of binary images
   ######################################################3 seperate_obj_x_thresh
    # Apply Gaussian blur to reduce noise
    #blurred = cv2.GaussianBlur(img, (7, 7), 1.2)

    # Threshold the image to convert it to a binary image
    _, thresh = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour and create a separate binary image for each object
    objects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        obj = img[y:y + h, x:x + w].copy()
        obj_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(obj_mask, [contour-np.array([x, y])], -1, 1, -1)
        obj[obj_mask == 0] = 0
        if sum(sum(obj)) < seperate_obj_x_thresh*w*h: ################## another i dot filter ###############
            objects.append((x, y, obj))
    # Sort the objects based on their X position in the original image
    objects.sort(key=lambda x: x[0])

##    # This part is trying to avoid deviation of letter for two:
    # mounted_objects = []
    # two_times_only = True
    # for i in range(len(objects)):
    #     x = objects[i][0]
    #     y = objects[i][1]
    #     obj = objects[i][2]
    #     h, w = obj.shape[:2]
    #     try:
    #         overlap = check_overlap(x, w, x_pre, w_pre)
    #         if (w == overlap or overlap == w_pre) and two_times_only:
    #             mounted_img = obj_mount(obj, objects[i-1][2], x, y, objects[i-1][0], objects[i-1][1])
    #             mounted_objects[-1] = (min(x, x_pre), mounted_img)
    #             two_times_only = False
    #         else:
    #             mounted_objects.append((x, obj))
    #             x_pre = x
    #             w_pre = w
    #             y_pre = y
    #             two_times_only = True
    #     except:
    #         mounted_objects.append((x, obj))
    #         x_pre = x
    #         w_pre = w
    #         y_pre = y
    #         two_times_only = True
    #    plt.imshow(mounted_objects[-1][1])
    #    plt.title("img")
        #plt.show()
    # return [obj for (x, obj) in mounted_objects]

    return [obj for (x,y, obj) in objects]

def check_overlap(x1,w1,x2,w2):
    ## This function returns the X axis overlap between two binary images
    ## inputs: x1,x2 - x start point of image
    ##         w1,w2 - width of each image
    ## output: the overlap

    if x1+w1 < x2 or x2+w2 < x1 :
        return 0
    else:
        if (x1 < x2 and x2 + w2 < x1 + w1):
            return w2
        elif (x2 < x1 and x1 + w1 < x2 + w2):
            return w1
        else:
            x_min_ov = max(x1,x2)
            x_max_ov = min(x1+w1,x2+w2)
            overlap = abs(x_max_ov-x_min_ov)
        return overlap

def obj_mount(image1, image2, x1, y1, x2, y2):
    ## This function is mounting two binary images to one
    ## input: image1,image2 - binary images
    ##        x1,y1,x2,y2 - coordinates of each image starting point

    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    height = max(height1 + y1, height2 + y2)
    width = max(width1 + x1, width2 + x2)
    combined_image = np.zeros((height, width), dtype=np.uint8)
    combined_image[y1:y1 + height1, x1:x1 + width1] = image1
    combined_image[y2:y2 + height2, x2:x2 + width2] = image2
   # plt.imshow(combined_image)
   # plt.title("img")
    #plt.show()
    # #combined_image = crop_interesting_part(combined_image,41)
    # plt.imshow(combined_image)
    # plt.title("interest")
    # plt.show()
    return combined_image

def convert_label_to_letter(label,i):
    if i == 0:
        letter_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    else:
        letter_labels = "0123456789abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    return letter_labels[label]

def convert_label_to_letter2(label):
    letter_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return letter_labels[label]

def predict_image(image):
    #disable the output of the predict function
    global loaded_model
    K.set_learning_phase(0)
    prediction = loaded_model.predict(image*255)
    return prediction

def find_letters(X,eps_letters,min_samps_letters,sigma,i_dot_thresh, x_pad, y_pad, seperate_obj_x_thresh):
    """ Get as an input an image of a word, and seperate them to letters,
    and preform prediction on each letter"""
    groups = separate_objects(X, seperate_obj_x_thresh)
    string = ""
    for i in range(len(groups)):
     #   plt.imshow(groups[i], cmap='gray')
     #   plt.title("groups[i]")
        #plt.show()
        vec_to_model = pre_process_to_emnist(groups[i],i_dot_thresh, x_pad, y_pad)
        if vec_to_model is not 0:
            # vec_to_model = 1-vec_to_model
            #filted = gaussian_filter(vec_to_model, sigma=sigma)

      #      plt.imshow(vec_to_model, cmap='gray')
    #        plt.title("filted[i]")
            #plt.show()

            current_letter = convert_label_to_letter(np.argmax(predict_image(vec_to_model.reshape((1,784)))),i)
            transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.5,), (0.5,))
            ])
            image = transform(vec_to_model).unsqueeze(0)

            # make the prediction
            image = image.float()
            #output = model(image)
           # output = output.detach().numpy()
            # current_letter_model_2 = convert_label_to_letter2(np.argmax(output))
            #print(current_letter_model_2)
           # plt.imshow(vec_to_model.reshape(28, 28), cmap='gray')
            current_letter = num_to_letter(current_letter)
           # plt.title("I have predicted: " + current_letter)
            #plt.show()
            string += current_letter


    # print("The real word is: " + string)
    # print("I HAVE PREDICTED: " + Speller(lang='en')(string))
    return string
    #return Speller(lang='en', fast=True)(string)

def find_sequence(X, min_samps_letters, eps_letters, min_samps_words, eps_words, sigma,i_dot_thresh, x_pad, y_pad, seperate_obj_x_thresh):
    """ Get as an input an image of a sequence of words, and seperate them to words,
    and preform prediction on each word by calling find_words"""
    global canvas_1, window
    temp_list = find_1(X)
    #plot the image
  #  plt.imshow(X, cmap='gray')
  #  plt.title("X")
  #  plt.show()
    #cluster and return all the clusters from temp_list
    ### Do DBSCAN to cluster the words by density
    db = DBSCAN(eps=eps_words, min_samples=min_samps_words).fit(temp_list)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # DEBUGING PURPOSE
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    groups = []
    width, height = len(X[0]), len(X)

    # Iterate over the clusters
    for k, col in zip(unique_labels, colors):
        # Get the indices of the points that are part of the cluster
        indices = np.where(labels == k)[0]
        # Use the indices to get the points from temp_list
        xy = np.array(temp_list)[indices]
        #create a zeros matrix in the size of X and put 1 in the locations from xy
        temp = np.zeros((height, width))
        for i in range(len(xy)):
            temp[xy[i][0]][xy[i][1]] = 1
        groups.append((temp, (min(xy[:, 0]), max(xy[:, 0]), min(xy[:, 1]), max(xy[:, 1]))))
    #plot all the Bounding boxes of the words on the original image
    # Create a Matplotlib figure
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    # Display the image on the axis
    ax.imshow(X)

    # Add each bounding box to the axis as a rectangle patch
    for box in groups[:-1]:
        ymin, ymax, xmin, xmax = box[1]
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Update the existing canvas_1 with the new figure
    canvas_1 = FigureCanvasTkAgg(fig, master=window)
    #canvas_1.get_tk_widget().grid(row=0, column=0)
    canvas_1.draw()
    canvas_1.get_tk_widget().grid(row=0, column=1, rowspan=8)

    # Show the final image with bounding boxes

    # initiating the loop, the fitst iteration is a bit different so executed seperately:
    temp = groups.pop(find_closest_square(groups))
    temp_x,temp_y = temp
  #  plt.imshow(temp[0],cmap='gray')
    # plt.show()
    final_str = ""
    mama_mia = crop_to_word(temp_x) # consider using the BB achieved from DBScan
    current_word = find_letters(mama_mia,eps_letters,min_samps_letters,sigma,i_dot_thresh, x_pad, y_pad, seperate_obj_x_thresh)
    final_str += current_word + " "
  #  plt.imshow(mama_mia, cmap='gray')
  #  plt.title("I have predicted: " + current_word)
 #   plt.show()

    ##here we need to do the same thing as in find_words, but with the groups other words
    for i in range(len(groups)):
        try:
            temp_x,temp_y = find_closest_square_in_x(temp_y, groups)
         #   plt.imshow(temp_x, cmap='gray')
            # plt.show()
            mama_mia = crop_to_word(temp_x)
            current_word = find_letters(mama_mia, eps_letters, min_samps_letters, sigma,i_dot_thresh, x_pad, y_pad, seperate_obj_x_thresh)
            final_str += current_word + " "
          #  plt.imshow(mama_mia, cmap='gray')
          #  plt.title("prediction: " + current_word)
         #   plt.show()
            #remove from groups the element where the first index is the temp_x
            for j in range(len(groups)):
                if temp_y == groups[j][1]:
                    temp_x,temp_y = groups.pop(j)
                    break
        except:
            # print("MOVED A NEW LINE")
            temp = groups.pop(find_closest_square(groups))
            temp_x,temp_y = temp
            final_str += find_letters(crop_to_word(temp_x),eps_letters,min_samps_letters,sigma,i_dot_thresh, x_pad, y_pad, seperate_obj_x_thresh) + " "
          #  plt.imshow(crop_to_word(temp_x))
        #    plt.title("prediction: " + find_letters(crop_to_word(temp_x),eps_letters,min_samps_letters,sigma,i_dot_thresh, x_pad, y_pad))
            #plt.show()
    # print("I HAVE PREDICTED: " + final_str)
    return final_str

def complete_saved_vid2text(gray_frame,parameters):
    global pred_label, before_speller
    global window
    # Binarization:
    Thresh = parameters["Thresh"]
    _, binary = cv2.threshold(gray_frame, Thresh, 255, cv2.THRESH_BINARY) # apply thresholding
  #  cv2.imshow('Binary image form video', gray_frame)
  #  cv2.waitKey(0)
  #  cv2.imshow('Binary image form video', (255-binary)/255)
  #  cv2.waitKey(0)

    # Set parameters:
    min_lets = parameters.get("min_lets")
    eps_lets = parameters.get("eps_lets")
    min_words = parameters.get("min_words")
    eps_words = parameters.get("eps_words")
    sigma = parameters.get("sigma")
    x_pad = parameters.get("x_pad")
    y_pad = parameters.get("y_pad")
    i_dot_thresh = parameters.get("i_dot_thresh")
    seperate_obj_x_thresh = parameters.get("seperate_obj_x_thresh")

    # min_lets = 40
    # eps_lets = 1
    # min_words = 15
    # eps_words = 20
    # sigma = 0
    # x_pad = 3
    # y_pad = 3
    # i_dot_thresh = 35

    # Text:
    print("Min letters: ", min_lets)
    print("Eps letters: ", eps_lets)
    print("Min words: ", min_words)
    print("Eps words: ", eps_words)
    print("Sigma: ", sigma)
    print("X pad: ", x_pad)
    print("Y pad: ", y_pad)
    print("I dot thresh: ", i_dot_thresh)
    print("Seperate obj x thresh: ", seperate_obj_x_thresh)

    final_strings = (find_sequence((255-binary)/255, min_lets, eps_lets, min_words, eps_words, sigma, i_dot_thresh, x_pad, y_pad,  seperate_obj_x_thresh))

    # Print predictions:
    before_speller.config(text=final_strings)
    print(final_strings)
    print("After Speller:")
    print(Speller(lang='en', fast=True)(final_strings))
    pred_label.config(text=Speller(lang='en', fast=True)(final_strings))
    window.update()
    text_file = open("final_strings.txt", "w")
    # n = text_file.write(str(final_strings))
    # text_file.close()



def find_biggest_rect(gray_image,paramters):
    min_area = paramters["min_area"]
    pixels_from_edge = paramters["pixels_from_edge"]

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection to obtain edges
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Find contours in the edge image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list to store rectangles
    rectangles = []

    # Loop over all contours
    for contour in contours:
        # Approximate contour as polygon
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Check if polygon has four sides and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Compute the bounding box of the polygon
            rect = cv2.boundingRect(approx)

            # Check if bounding box is rectangular
            aspect_ratio = float(rect[2]) / rect[3]
            if 0.5 <= aspect_ratio <= 2.0:
                # Add the rectangle to the list
                rectangles.append(rect)

    # Sort rectangles in descending order of area
    rectangles = sorted(rectangles, key=lambda x: x[2]*x[3], reverse=True)

    # If no rectangles are found, return -1
    if len(rectangles) == 0:
        return -1

    if (rectangles[0][2]*rectangles[0][3]) < min_area:
        return -1

    # Get the biggest rectangle
    biggest_rect = rectangles[0]

    # Crop the image inside the biggest rectangle and cut 1% of the borders to remove noise
    x, y, w, h = biggest_rect
    x1 = int(x + pixels_from_edge/100 * w)
    y1 = int(y + pixels_from_edge/100 * h)
    x2 = int(x + (1-pixels_from_edge/100) * w)
    y2 = int(y + (1-pixels_from_edge/100) * h)
    cropped = gray_image[y1:y2, x1:x2]
    #    cropped_img = gray_image[biggest_rect[1]:biggest_rect[1]+biggest_rect[3], biggest_rect[0]:biggest_rect[0]+biggest_rect[2]]

    return cropped


def mp4_to_text(video_path):
    # Load the video from video_path:
    cap = cv2.VideoCapture(video_path)
    # Check if camera opened successfully

    # Check if the video capture was successfully opened
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frame rate (frames per second) of the video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the number of frames to skip to get a new frame every 5 seconds
    frames_to_skip = int(frame_rate * 40)

    # Set the starting frame number
    current_frame = 0

    # Loop through each frame in the video
    while current_frame < total_frames:
        # Read the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)  # set the position to the 53rd frame
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Whiteboard Text", gray)
        # cv2.waitKey(0)
        white_image = np.sum(gray)
        [w, h] = gray.shape[:]
        image_area = w * h
        if white_image >= image_area * 255 * .6:
            # plot the frame and wait 1 sec and than close it
            cv2.imshow("Whiteboard Text", gray)
            temp = find_biggest_rect(gray)
            if temp is not -1:
                plt.imshow(temp)
                plt.show()
                current_frame += 15
                continue
            else:
                current_frame += 15
                continue
            cv2.imshow("Whiteboard Text", frame)
            cv2.waitKey(0)

            # Save the cropped frame to file
            cv2.imwrite("whiteboard_text_{}.jpg".format(current_frame), frame)

            # Increment the current frame number
            current_frame += frames_to_skip

            # Skip the next `frames_to_skip` frames
            for i in range(frames_to_skip):
                cap.read()
        else:
            current_frame += 30

    # Release the video capture
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


def update_params(param_name, new_value):
    parameters[param_name] = new_value


def print_params():
    print(parameters)

def change_flag():
    global flag
    flag = not flag
def live_webcam_stream(parameters):
    global window
    global live
    # global eps_words, eps_words_label, pixels_from_edge_entry, pixels_from_edge_label, min_area_entry, \
    #     min_area_label, min_lets_entry, min_lets_label, eps_lets_entry, eps_lets_label, \
    #     min_words_entry, min_words_label, sigma_entry, sigma_label, x_pad_entry, x_pad_label, \
    #     y_pad_entry, y_pad_label, i_dot_thresh_entry, i_dot_thresh_label
    #
    # def bind_params(entry_widget, param_name):
    #     var = tk.StringVar()
    #     var.trace("w", lambda name, index, mode, var=var: update_params(param_name, var.get()))
    #     entry_widget.config(textvariable=var)
    #
    # flag = False
    # # add entries to update the parameters
    # eps_words = tk.Entry(window, width=10)
    # eps_words.grid(column=3, row=0)
    # eps_words.insert(0, parameters['eps_words'])
    # eps_words_label = tk.Label(window, text="Epsilon For DBScan")
    # eps_words_label.grid(column=2, row=0)
    # bind_params(eps_words, 'eps_words')
    #
    # pixels_from_edge_entry = tk.Entry(window, width=10)
    # pixels_from_edge_entry.grid(column=3, row=1)
    # pixels_from_edge_entry.insert(0, parameters['pixels_from_edge'])
    # pixels_from_edge_label = tk.Label(window, text="Pixels From Edge")
    # pixels_from_edge_label.grid(column=2, row=1)
    # bind_params(pixels_from_edge_entry, 'pixels_from_edge')
    #
    # min_area_entry = tk.Entry(window, width=10)
    # min_area_entry.grid(column=3, row=2)
    # min_area_entry.insert(0, parameters['min_area'])
    # min_area_label = tk.Label(window, text="Min Area")
    # min_area_label.grid(column=2, row=2)
    # bind_params(min_area_entry, 'min_area')
    #
    # min_lets_entry = tk.Entry(window, width=10)
    # min_lets_entry.grid(column=3, row=3)
    # min_lets_entry.insert(0, parameters['min_lets'])
    # min_lets_label = tk.Label(window, text="Min Letters")
    # min_lets_label.grid(column=2, row=3)
    # bind_params(min_lets_entry, 'min_lets')
    #
    # eps_lets_entry = tk.Entry(window, width=10)
    # eps_lets_entry.grid(column=3, row=4)
    # eps_lets_entry.insert(0, parameters['eps_lets'])
    # eps_lets_label = tk.Label(window, text="Epsilon For DBScan")
    # eps_lets_label.grid(column=2, row=4)
    # bind_params(eps_lets_entry, 'eps_lets')
    #
    # min_words_entry = tk.Entry(window, width=10)
    # min_words_entry.grid(column=3, row=5)
    # min_words_entry.insert(0, parameters['min_words'])
    # min_words_label = tk.Label(window, text="Min Words")
    # min_words_label.grid(column=2, row=5)
    # bind_params(min_words_entry, 'min_words')
    #
    # sigma_entry = tk.Entry(window, width=10)
    # sigma_entry.grid(column=3, row=6)
    # sigma_entry.insert(0, parameters['sigma'])
    # sigma_label = tk.Label(window, text="Sigma")
    # sigma_label.grid(column=2, row=6)
    # bind_params(sigma_entry, 'sigma')
    #
    # x_pad_entry = tk.Entry(window, width=10)
    # x_pad_entry.grid(column=3, row=7)
    # x_pad_entry.insert(0, parameters['x_pad'])
    # x_pad_label = tk.Label(window, text="X Padding")
    # x_pad_label.grid(column=2, row=7)
    # bind_params(x_pad_entry, 'x_pad')
    #
    # y_pad_entry = tk.Entry(window, width=10)
    # y_pad_entry.grid(column=3, row=8)
    # y_pad_entry.insert(0, parameters['y_pad'])
    # y_pad_label = tk.Label(window, text="Y Padding")
    # y_pad_label.grid(column=2, row=8)
    # bind_params(y_pad_entry, 'y_pad')
    #
    # i_dot_thresh_entry = tk.Entry(window, width=10)
    # i_dot_thresh_entry.grid(column=3, row=9)
    # i_dot_thresh_entry.insert(0, parameters['i_dot_thresh'])
    # i_dot_thresh_label = tk.Label(window, text="I Dot Threshold")
    # i_dot_thresh_label.grid(column=2, row=9)
    # bind_params(i_dot_thresh_entry, 'i_dot_thresh')
    #
    #
    # #add a print paramters button to the window
    # # print_params_button = tk.Button(window, text="Print Parameters")
    # # print_params_button.grid(row=15, column=5)
    #
    # #create button that will stop the main loop and will able us to change the parameters
    # stop_button = tk.Button(window, text="Stop", command=change_flag)
    # stop_button.grid(row=15, column=5)
  #  button = tk.Button(window, text="Update Parameters", command=lambda: update_parameters(parameters))
  #  button.grid(column=2, row=10)
    #connect to my iphone camera and stream the video
    cap = cv2.VideoCapture(0)
    # Define the range of white color in HSV
    lower = np.array([0, 0, 100], dtype="uint8")
    upper = np.array([179, 50, 255], dtype="uint8")
    cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)
    start_time = time.time()
    all_the_drames = []
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        # Check if the frame was successfully read
        if not ret:
            #sleep for 1ms
            continue
        #show the frame in the window at frame

        cv2.imshow("Live Video", frame)

        # Check if the current frame is the first frame or if 5 seconds have passed
        if ret == True and (time.time() - start_time) >= parameters['time_between_frames']:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("Whiteboard Text", gray)
            # cv2.waitKey(0)

            gray = find_biggest_rect(gray,parameters)
            if gray is not -1:
            #    plt.imshow(gray)
            #    plt.show()
            #    continue
                white_image = np.sum(gray)
                [w, h] = gray.shape[:]
                image_area = w*h
                #if we want to set the רף of the image to be white
               # raph = image_area*255*0.65
               # if white_image >= raph:                # Crop the whiteboard region
                print("Begin Process ------->")
              #  cv2.imshow("Whiteboard Text", gray)
               # cv2.waitKey(0)
                #make the waitkey 0 or press any key to continue
                """min_samps_letters, eps_letters, min_samps_words, eps_words, sigma, i_dot_thresh,
                                        x_pad, y_pad"""
                complete_saved_vid2text(gray,parameters)
                # Show the cropped frame
                # Reset the start time
                start_time = time.time()
                all_the_drames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the windows
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    parameters = {
        "pixels_from_edge": 7,
        "time_between_frames": 2,
        "min_area": 18000,
        "min_lets": 40,
        "eps_lets": 1,
        "min_words": 15,
        "eps_words": 3,
        "sigma": 0,
        "x_pad": 3,
        "y_pad": 3,
        "i_dot_thresh": 35,
        "Thresh": 135,
        'seperate_obj_x_thresh': 0.65
    }
    # load the model:
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")


    # create the window
    window = tk.Tk()
    window.title("Summarizer")
    window.geometry("1200x600")


    # Create a Tkinter canvas to display the Matplotlib figure

    # add a button to the window and set the text and
    button = tk.Button(window, text="Start Live Stream", command=lambda: live_webcam_stream(parameters))
    button.grid(column=0, row=0)
    text = "Waiting for the user to press the button"
    # generate a label and update it every time the label variable changes
    label_before_speller = tk.Label(window, text="Before Speller")
    label_before_speller.grid(column=0, row=1,sticky="n")
    before_speller = tk.Label(window, text="Waiting for the user to press the button")
    before_speller.grid(column=0, row=2,sticky="n")
    label = tk.Label(window, text="The Predicted Text is: ")
    label.grid(column=0, row=3, sticky="n")
    pred_label = tk.Label(window, text=text)
    pred_label.grid(column=0, row=4, sticky="n", rowspan=5)
    #move all the labels to the center
    window.grid_columnconfigure(0, weight=1)
    #move before_speller and label to the center and increase the font size and set it to Tahoma
    button.configure(font=("Tahoma", 40))
    before_speller.grid(sticky="nsew")
    before_speller.config(font=("Tahoma", 40))
    label_before_speller.grid(sticky="nsew")
    label_before_speller.config(font=("Tahoma", 40))
    label.grid(sticky="nsew")
    label.config(font=("Tahoma", 40))
    pred_label.grid(sticky="nsew")
    pred_label.config(font=("Tahoma", 40))

    # add a button to the window and set the text and



    # window.grid_rowconfigure(0, weight=1)
    # window.grid_rowconfigure(1, weight=1)
    # window.grid_rowconfigure(2, weight=1)
    # window.grid_rowconfigure(3, weight=1)
    window.mainloop()

  #  live_webcam_stream(parameters)
#    mp4_to_text("vid.mp4")
    #
    # X = read_image("Board5.jpeg",135)
    # # X = read_image("Board.jpeg",120)
    # a = get_frames_from_video('Board_vid.mp4', 3)
    # # print(a)
    # cv2.imshow('hey', a[12])
    # cv2.waitKey(5000)
    # # Here we need a frame choosing algorithm
    # X = to_binary_image(a[12],135)
    #
    # # plt.imshow(X)
    # # plt.show()
    # min_lets=40
    # eps_lets=1
    # min_words=15
    # eps_words=20
    # sigma = 0
    # x_pad = 3
    # y_pad = 3
    # i_dot_thresh = 35
    # final_strings = ((find_sequence(X, min_lets, eps_lets, min_words, eps_words, sigma, i_dot_thresh, x_pad, y_pad)))
    # print(final_strings)
    # print("After Speller:")
    # print(Speller(lang='en', fast=True)(final_strings))
    # text_file = open("final_strings.txt", "w")
    # n = text_file.write(str(final_strings))
    # text_file.close()
    #
    # # final_strings = []
    # # for eps_lets in range(29, 30):
    # #     for min_lets in range(20, 21):
    # #         for eps_words in range(20, 21):
    # #             for min_words in range(15, 16):
    # #                 for sigma in range(4, 5):
    # #                     print(sigma)
    # #                     final_strings.append((find_sequence(X, min_lets, eps_lets/10, min_words, eps_words, 8/20), min_lets, eps_lets/10, min_words, eps_words,sigma/10))

  