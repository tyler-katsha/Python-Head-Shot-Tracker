import cv2 as cv
import modules.FaceMeshModule as fmm
import time
wCam,hCam = 1080,980
cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
blue = (255,0,0)
red = (0,0,255)
black = (0,0,0)
green = (0,255,0)
yellow = (0,255,255)
detector = fmm.FaceMeshDetector(maxFaces=1,minDetectionCon=0.85,minTrackCon=0.85)

#print(len(detector.faces))
while True:
  success,img = cap.read()
  img,faces = detector.findFaceMesh(img,draw=False)
  #Checking for faces
  if faces:
    face = faces[0]
    #Target points choose between 9 or 151
    tid,tx,ty = face[151]

    if len(face) != 0:
      #print(f'Target Found')   

      #Draw a circle at point 9 or 151
      
      #print(f'Target Landmark {tid}: x {tx}, y {ty}')
      cv.circle(img,(tx,ty),10,red,cv.FILLED)
      #Draw horizontal and vertical line
      cv.line(img,(0,ty),(wCam,ty),(0,0,0),1)      
      cv.line(img,(tx,0),(tx,hCam),(0,0,0),1)
      cv.circle(img,(tx,ty),18,red,1)
      cv.putText(img,str([tx,ty]),( tx+15 , ty-15),cv.FONT_HERSHEY_PLAIN,1,yellow,2)

      #Text info
      cv.putText(img,f'Target Aquired',(4,80),cv.FONT_HERSHEY_PLAIN,1.7,green,2)
      cv.putText(img,f'People found: {str(int(len(faces)))}',(4,110),cv.FONT_HERSHEY_PLAIN,1.7,green,2)
      cv.putText(img,f'Coords: {str([tx,ty])}',(4,140),cv.FONT_HERSHEY_PLAIN,1.7,yellow,2)
      
  else:
    #print("Locating target...")
    cv.putText(img,f'Locating...',(4,80),cv.FONT_HERSHEY_PLAIN,1.7,red,2)
    cv.putText(img,f'People found: {str(0)}',(4,110),cv.FONT_HERSHEY_PLAIN,1.7,red,2)
    cv.putText(img,f'Coords: {str([0,0])}',(4,140),cv.FONT_HERSHEY_PLAIN,1.7,yellow,2)


  #FPS
  cTime = time.time()
  fps = 1/(cTime - pTime)
  pTime = cTime
  cv.putText(img,f'Fps:{str(int(fps))}',(4,50),cv.FONT_HERSHEY_PLAIN,1.7,blue,2)
  cv.imshow('Image',img)

  #Enter q to close window
  if cv.waitKey(1) & 0xFF in [ord('Q'), ord('q')]:
    cap.release()
    cv.destroyAllWindows()
