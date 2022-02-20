# import libraries
from ctypes import sizeof
import sys
import time
import numpy as np
print("pls work")
#sys.path.append('/s/bach/g/under/videep/.local/lib/python3.8/site-packages')
import cv2
import face_recognition

# Get a reference to webcam 
video_capture = cv2.VideoCapture("videos/g1.mp4")

# Initialize variables
frameNumber=0
face_locations = []
ret, prev_frame = video_capture.read()

p_frame_thresh = 900000
totalFaces = 0
numberOfKeyFrames = 0
name = ""
while ret:
    # Grab a single frame of video
    ret, curr_frame = video_capture.read()
     
    frameNumber = frameNumber+1
    if ret:
        diff = cv2.absdiff(curr_frame, prev_frame)  #compute the difference between frames 
        non_zero_count = np.count_nonzero(diff) 
        if non_zero_count > p_frame_thresh:         #checks if the current frame is unique
            rgb_frame = curr_frame[:, :, ::-1]      # Convert the image from BGR color
            # Find all the faces in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame, model="CNN")
            
            totalFaces = totalFaces + len(face_locations)
            numberOfKeyFrames = numberOfKeyFrames +1
            
            for top, right, bottom, left in face_locations:
                # Draw a box around the face
                cv2.rectangle(curr_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.imshow('Video', curr_frame)
            cv2.waitKey(1)

            if(len(face_locations) == 1):
                name = "1Face/Frame_" + str(frameNumber) + '.jpg'
                #cv2.imwrite(name, curr_frame)
            if(len(face_locations) == 2):
                name = "2Face/Frame_" + str(frameNumber) + '.jpg'
          #      cv2.imwrite(name, curr_frame)
            if(len(face_locations) == 3):
                name = "3Face/Frame_" + str(frameNumber) + '.jpg'
           #     cv2.imwrite(name, curr_frame)
            
            #cv2.imwrite('Frame_' + str(frameNumber) + '.jpg', curr_frame)
            prev_frame = curr_frame
print("Percent faces - " + str(totalFaces / (numberOfKeyFrames *3 )))
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()