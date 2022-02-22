# import libraries
from ctypes import sizeof
import subprocess
import sys
import time
import numpy as np
print("pls work")
#sys.path.append('/s/bach/g/under/videep/.local/lib/python3.8/site-packages')
import cv2
import face_recognition
# import glob


# for video in list(glob.glob("/Users/ilianacastillon/Research/videos/NXG/*.avi")):
#     print(video)

video = '/Users/ilianacastillon/Desktop/HBLGT.mp4'
# video = '/Users/ilianacastillon/Research/videos/NXG/G01_NXG.avi'

# Get a reference to webcam 
#fps = subprocess.call('ffmpeg -i ' + video + ' 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p"')
fps = subprocess.run(['ffmpeg','-i',video, '-n', '"s/.*, \(.*\) fp.*/\1/p"'])#/Users/ilianacastillon/Research/videos/NXG/G01_NXG.avi')
print(fps)
video_capture = cv2.VideoCapture(video)

# Initialize variables
frameNumber=0
face_locations = []
ret, prev_frame = video_capture.read()

#get fps from open cv
p_frame_thresh = 900000
totalFaces = 0
numberOfKeyFrames = 0
name = ""
oneCount = 0
twoCount = 0
threeCount = 0
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
                oneCount=oneCount+1
            if(len(face_locations) == 2):
                name = "2Face/Frame_" + str(frameNumber) + '.jpg'
                #cv2.imwrite(name, curr_frame)
                twoCount=twoCount+1
            if(len(face_locations) == 3):
                name = "3Face/Frame_" + str(frameNumber) + '.jpg'
               # cv2.imwrite(name, curr_frame)
                threeCount=threeCount+1
            #cv2.imwrite('Frame_' + str(frameNumber) + '.jpg', curr_frame)
            prev_frame = curr_frame

print("Percent one face detected - " + str(oneCount / (numberOfKeyFrames *3 )))
print("Percent two faces are detected - " + str(twoCount / (numberOfKeyFrames *3 )))
print("Percent three faces are detected - " + str(threeCount / (numberOfKeyFrames *3 )))
print("Percent of faces detected - " + str(totalFaces / (numberOfKeyFrames *3 )))
print(str(numberOfKeyFrames))

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()