# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import time
from tkinter import * 
#import tkSimpleDialog
#import tkMessageBox
from tkinter import messagebox as tkMessageBox



#PREDICTOR_PATH = "/media/tpf/9CE2E75EE2E73B62/DATAStore/mayurDESK/Projects/Blink_to_speak_linux/shape_predictor_68_face_landmarks.dat"

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def eye_on_mask(mask, side):
    points = [landmarks[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)

    except:
        pass

LEFT_COUNTER = 0
RIGHT_COUNTER = 0
UP_COUNTER = 0
DOWN_COUNTER = 0

def center_calc(thresh):
    global LEFT_COUNTER,RIGHT_COUNTER,UP_COUNTER,DOWN_COUNTER
    try:
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key = cv2.contourArea)
        center = cv2.moments(cnt)
        for k in cnt :
            #print(k)
            if 50 < k[0][0] and 62 > k[0][0]:
                #print('left')
                LEFT_COUNTER += 1
            elif 30 < k[0][0] and 36 > k[0][0]:
                #print('right')
                RIGHT_COUNTER += 1
            elif 170 < k[0][1] and 180 > k[0][1]:
                #print('up')
                UP_COUNTER += 1
            elif 198 < k[0][1] and 205 > k[0][1]:
                #print('down')
                DOWN_COUNTER += 1

    except:
        pass

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

#cap = cv2.VideoCapture(0)


cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
cv2.setTrackbarPos('threshold', 'image', 86)


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="/media/tpf/9CE2E75EE2E73B62/DATAStore/mayurDESK/Projects/Blink_to_speak_linux/shape_predictor_68_face_landmarks.dat")
'''ap.add_argument("-v", "--video", type=str, default="",
    help="path to input video file")'''
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
#predictor = dlib.shape_predictor(PREDICTOR_PATH)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#print(lStart,lEnd)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#print(rStart,rEnd)

# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

   
FULL_POINTS = list(range(0, 68))  
FACE_POINTS = list(range(17, 68))  
   
JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68)) 
   
#EYE_AR_THRESH = 0.25  
#EYE_AR_CONSEC_FRAMES = 3  
   
COUNTER_LEFT = 0  
TOTAL_LEFT = 0  
   
COUNTER_RIGHT = 0  
TOTAL_RIGHT = 0

total_time_res=0
total_time=0


root = Tk()
w = Label(root,text='Blink to Speak')
w.pack()
    
# loop over frames from the video stream
while True:

    start_time_res=time.time()
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    #frame = vs.read()
    ret, img = vs.read()
    thresh = img.copy()
    #frame = imutils.resize(frame, width=450)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = imutils.resize(img,width=450)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
        # loop over the face detections

    #ret, img = cap.read()
    
    #rects = detector(gray, 1)

    for rect in rects:
        x = rect.left()  
        y = rect.top()
        x1 = rect.right()  
        y1 = rect.bottom()  

        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])  
    
        left_eye = landmarks[LEFT_EYE_POINTS]  
        right_eye = landmarks[RIGHT_EYE_POINTS]  
    
        left_eye_hull = cv2.convexHull(left_eye)  
        right_eye_hull = cv2.convexHull(right_eye)  
        cv2.drawContours(img, [left_eye_hull], -1, (0, 255, 0), 1)  
        cv2.drawContours(img, [right_eye_hull], -1, (0, 255, 0), 1)  
    
        ear_left = eye_aspect_ratio(left_eye)  
        ear_right = eye_aspect_ratio(right_eye)    
    
        if ear_left < EYE_AR_THRESH:  
            COUNTER_LEFT += 1  
        else:  
            if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:  
                TOTAL_LEFT += 1  
                #print("Left eye winked")  
            COUNTER_LEFT = 0  
    
        if ear_right < EYE_AR_THRESH:  
            COUNTER_RIGHT += 1  
        else:  
            if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:  
                TOTAL_RIGHT += 1  
                #print("Right eye winked")  
            COUNTER_RIGHT = 0

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        #print(leftEye,rightEye)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        


        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        #ear= int(ear)
        #print(ear)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                #SHUT_EYE_COUNTER += 1
            # reset the eye frame counter
            COUNTER = 0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        
        
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        
        '''start_time=time.time()
        
        stop_time=time.time()
        time_diff=stop_time-start_time
        total_time+=time_diff
        if total_time >= 2 :'''
            

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        #print(mid)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        contouring(thresh[:, 0:mid], mid, img)
        contouring(thresh[:, mid:], mid, img, True)
        center_calc(thresh[:, mid:])
        
             

        end_time_res=time.time()

        time_diff_res=end_time_res-start_time_res

        total_time_res +=time_diff_res
        #print(total_time_res)

        if total_time_res >= 3 :

            Counter=[LEFT_COUNTER,RIGHT_COUNTER, UP_COUNTER, DOWN_COUNTER]
            direction=(max(Counter))
            res_index=Counter.index(direction)

            if res_index == 0 :
                print('left')
                total_time_res=0
                LEFT_COUNTER += 1
                
            elif res_index == 1 :
                print('right')
                total_time_res=0
                RIGHT_COUNTER += 1

            elif res_index == 2 :
                print('up')
                #text='Toilet'
                #cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                total_time_res=0
                UP_COUNTER += 1

            elif res_index == 3 :
                print('down')
                total_time_res=0
                DOWN_COUNTER += 1
            
            if TOTAL == 1:
                #if total_time_res >= 3:
                text='yes'
                #tkMessageBox.showinfo('Blink Output',text)
                #cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)
                total_time_res=0
                TOTAL=0

            elif TOTAL == 2:
                #if total_time_res >= 3:
                text='no'
                #cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)
                total_time_res=0
                TOTAL=0

            elif TOTAL ==  3:
                #if total_time_res >= 3:
                text="I'm Okay"
                #cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)
                total_time_res=0
                TOTAL=0

            elif TOTAL ==  4:
                #if total_time_res >= 3:
                text="I want to sleep"
                #cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0

            elif TOTAL > 4:
                text="Wrong"
                #cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0

            elif TOTAL == 1 and LEFT_COUNTER == 1 :
                text="Call Guardian"
                #cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0
                LEFT_COUNTER=0

            elif TOTAL == 1 and RIGHT_COUNTER == 1 :
                text="Call Doctor"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0
                RIGHT_COUNTER=0
            
            elif LEFT_COUNTER == 1:
                text="Breathlessness"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                LEFT_COUNTER=0

            elif UP_COUNTER == 1:
                text="Toilet"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                UP_COUNTER=0
            
            elif RIGHT_COUNTER == 1 and LEFT_COUNTER == 1 :
                text="Water"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0
                LEFT_COUNTER=0
                RIGHT_COUNTER=0

            elif TOTAL == 1 and UP_COUNTER == 1 :
                text="Heartache"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0
                UP_COUNTER=0

            elif TOTAL_LEFT > 3:
                text="Emergency"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL_LEFT=0

            elif TOTAL_LEFT == 2:
                text="I have a problem"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL_LEFT=0
            
            elif TOTAL_LEFT == 3:
                text="I love you"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL_LEFT=0

            elif TOTAL_RIGHT == 1 and TOTAL_LEFT == 1 and TOTAL == 2:
                text="Thank You"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0
                TOTAL_LEFT=0
                TOTAL_RIGHT=0

            elif UP_COUNTER == 1 and DOWN_COUNTER == 1 and TOTAL == 2:
                text="I need a hug"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                UP_COUNTER=0
                DOWN_COUNTER=0
                TOTAL=0

            elif UP_COUNTER == 1 and DOWN_COUNTER == 1 and TOTAL_LEFT == 1:
                text="Happy"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                UP_COUNTER=0
                DOWN_COUNTER=0
                TOTAL_LEFT=0

            elif TOTAL == 2 and TOTAL_LEFT == 2 :
                text="Adjust"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0
                TOTAL_LEFT=0
            
            elif TOTAL == 1 and TOTAL_LEFT == 2 :
                text="Change"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0
                TOTAL_LEFT=0

            elif TOTAL_LEFT == 2 and TOTAL_RIGHT == 2 :
                text="Scratch"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL_LEFT=0
                TOTAL_RIGHT=0

            elif TOTAL_LEFT == 1 and TOTAL_RIGHT == 1 and TOTAL == 1 :
                text="Wash"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0
                TOTAL_RIGHT=0
                TOTAL_LEFT=0

            elif UP_COUNTER == 1 and TOTAL == 2 :
                text="Lift"
                cv2.putText(img, "OUTPUT : {}".format(text), (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(text)  
                total_time_res=0
                TOTAL=0
                UP_COUNTER=0
            
            else:
                break
            total_time_res=0
            TOTAL=0
            

        cv2.putText(img, "E.A.R. Left : {:.2f}".format(ear_left), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  
        cv2.putText(img, "E.A.R. Right: {:.2f}".format(ear_right), (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Wink Left : {}".format(TOTAL_LEFT), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  
    cv2.putText(img, "Wink Right: {}".format(TOTAL_RIGHT), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    
    
        


        #for (x, y) in shape[36:48]:
            #cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # show the image with the face detections + facial landmarks	

            # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
    cv2.putText(img, "Blinks: {}".format(TOTAL), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #cv2.putText(img, "Shuts: {}".format(SHUT_EYE_COUNTER), (10, 50),
            #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)		
    

    l = cv2.rectangle(img,(220,150),(480,200),(250,0,0),2)
    # show the frame
    cv2.imshow("Frame", l)
    cv2.imshow("image", thresh)
    key = cv2.waitKey(1) & 0xFF


    if key == ord('q'):
        break
vs.release()
cv2.destroyAllWindows()
vs.stop()




