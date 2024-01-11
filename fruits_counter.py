
import os
import sys
# sys.path.append("G:/DS_Interview_Prep/Project/Computer_vision/People_counter/yolov5peoplecounterwin11-main")

import cv2
import torch
from tracker import *
import numpy as np
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load("ultralytics/yolov5","custom", path="g:/DS_Interview_Prep/Project/Computer_vision/Fruits counting/best.pt")

tracker =Tracker()

area= [123,217,766,217]
cap=cv2.VideoCapture("g:/DS_Interview_Prep/Project/Computer_vision/Fruits counting/istockphoto-837639466-640_adpp_is.mp4")

ids=set()

while True:

   ret, frame=cap.read()

   frame=cv2.resize(frame,(800,500))
   cv2.line(frame,(123,217),(766,217),(0,255,244),3)
   result=model(frame)

   list = []
   for index, rows in  result.pandas().xyxy[0].iterrows():
#         print(rows)
#         print(index)
        x1=int(rows['xmin'])
        y1=int(rows['ymin'])
        x2=int(rows['xmax'])
        y2=int(rows['ymax'])
        name=str(rows['name'])
        if "Apples" in name: 
            list.append([x1,y1,x2,y2])

   boxes_id=tracker.update(list) #to assign id to ech boxes or rectangles 
   
   for box_id in boxes_id:
      x,y,w,h,id =box_id 
      cx, cy = (x+w)//2, (y+h)//2
      cv2.rectangle(frame,(x,y),(w,h),(0,255,0),2)
      cv2.putText(frame,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3) 
      cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
      if area[0]< cx < area[2] and area[1]-15 < cy <area[1]+15:
         ids.add(id)
         cv2.line(frame,(123,217),(766,217),(0,255,0),3)

   d=len(ids)
   cv2.putText(frame,str(d),(30,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
   cv2.imshow("Frame",frame)





   if cv2.waitKey(1) & 0xFF == ord("q"):
      break
        
cap.release()
cv2.destroyAllWindows()





   
   
    

