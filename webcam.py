#!/usr/bin/env python2
import cv2
import os
import sys
import numpy as np
sys.path.append('.')
import tensorflow as tf
import detect_face
from mtcnn.mtcnn import MTCNN 

def main():
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    #video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture('sample2.mp4')
   
    #video_capture.set(3, 640)
    #video_capture.set(4, 480)
    
    minsize = 25 # minimum size of face
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
            print(boxes)
            for i in range(boxes.shape[0]):
                result = detect_features.detect_faces(img)
                pt1 = (int(boxes[i][0]), int(boxes[i][1]))
                pt2 = (int(boxes[i][2]), int(boxes[i][3]))
                cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0))

                for j in range(len(result)):
                        bounding_box = result[j]['box']
                        keypoints = result[j]['keypoints']
                        cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0))

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
                break

    video_capture.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
