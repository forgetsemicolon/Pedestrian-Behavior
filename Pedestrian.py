#!/usr/bin/env python2
import cv2 
import os
import sys
import numpy as np
sys.path.append('.')
import tensorflow as tf
import detect_face
import random
from mtcnn.mtcnn import MTCNN 

def speeds(obstacle):
    speed = []
    for i in range(60):
        a = random.randint(45, 80) 
        speed.append(round(a))
        speed.sort()
        # low is a list of speeds to slower

    #print(speed)
    #print(len(speed))
    if obstacle == False:
        return random.choice(speed)

    else:																
        return 0


def main():
    
    video_capture = cv2.VideoCapture('sample2.mp4')
   
    #video_capture.set(3, 640)
    #video_capture.set(4, 480)
    
    minsize =30 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    detect_features = MTCNN()



    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        while(True):
            ret, frame = video_capture.read()
            if not ret:
                break
            # Display the resulting frame
            img = frame[:,:,0:3]
            boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            
            for i in range(boxes.shape[0]):
                result = detect_features.detect_faces(img)
                pt1 = (int(boxes[i][0]), int(boxes[i][1]))
                pt2 = (int(boxes[i][2]), int(boxes[i][3]))
                
                
                
                
                if len(result) == 0:
                        print('no pedestrians')
                        a = speeds(obstacle=False)
                        dp1 =(100, 300)
                        dp2 =(400,600)
                        cv2.rectangle(frame, dp1, dp2, thickness=-1, color=(0, 255, 0))
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        num1 = 'Pedestrians: '+str(len(result))
                        cv2.putText(img,num1,(100,400), font, 1.0,(255,255,255),2,cv2.LINE_AA)
                        num2 = 'Speed: '+str(a)
                        cv2.putText(img,num2,(100,450), font, 1.0,(255,255,255),2,cv2.LINE_AA)

                else :
                        a = speeds(obstacle=True)
                        dp1 =(100, 300)
                        dp2 =(400,600)
                        cv2.rectangle(frame, dp1, dp2, thickness=-1, color=(0, 0, 255))
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        num1 = 'Pedestrians: '+str(len(result))
                        cv2.putText(img,num1,(100,400), font, 1.0,(255,255,255),2,cv2.LINE_AA)
                        num2 = 'Speed: '+str(a)
                        cv2.putText(img,num2,(100,450), font, 1.0,(255,255,255),2,cv2.LINE_AA)
                           

                for j in range(len(result)):
                        bounding_box = result[j]['box']
                        keypoints = result[j]['keypoints']
                        cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0))
                        print(boxes)
                        cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
                        cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
                        cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
                        cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
                        cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)
              
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            #cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0))
            cv2.resizeWindow('Video', 1800,900)
            cv2.imshow('Video', frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Ended Simulation")
                break

    video_capture.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
