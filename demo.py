# ---------------------------------------------------------
# DarkflowTensorRT
# Copyright (c) 2019
# Licensed under The MIT License [see LICENSE for details]
# Written by Rudy Nurhadi
# ---------------------------------------------------------

import os
import sys
import cv2
import time
import numpy as np

from DarkflowTensorRT import DarkflowTensorRT

CWD_PATH = os.path.dirname(os.path.realpath(__file__))

darkflowTensorRT = DarkflowTensorRT(os.path.join(CWD_PATH, 'model'), 'yolov2-tiny', 4, rebuild_engine=False)

cap = cv2.VideoCapture(0)

def main():
    while True:
        grabbed, frame = cap.read()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        if grabbed:
            frame = cv2.resize(frame, (640, 480))
            imgs = [frame] * 4 # try increasing this for fun
            now = time.time()
            predict_results = darkflowTensorRT.return_predict(imgs)
            print("Inference Time: %f" % (time.time() - now))
            for i, results in enumerate(predict_results):
                for result in results:
                    if result['confidence'] > 0.4:
                        top = result['topleft']['y']
                        left = result['topleft']['x']
                        bottom = result['bottomright']['y']
                        right = result['bottomright']['x']
                        label = result['label']
                        
                        if label == 'person':
                            cv2.rectangle(imgs[i], (left, top), (right, bottom), (255, 0, 0), thickness=2)
                        else:
                            cv2.rectangle(imgs[i], (left, top), (right, bottom), (0, 0, 255), thickness=2)
                            
                        cv2.putText(imgs[i], label, 
                                    (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 1)
                cv2.imshow('frame%d' % i, imgs[i])

    cv2.destroyAllWindows()
    cap.release()  
    
if __name__ == '__main__':
    main()
