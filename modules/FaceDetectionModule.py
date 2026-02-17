import cv2 as cv
import mediapipe as mp
import time

class faceDetector():
  def __init__(self,minDetectionCon=0.5):
    self.minDetectionCon = minDetectionCon
    
    self.mpFaceDetection = mp.solutions.face_detection
    self.mpDraw = mp.solutions.drawing_utils
    self.FaceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

  def findFaces(self,img,draw=True):
    self.imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    self.results = self.FaceDetection.process(self.imgRGB)

    bboxs = []
    if self.results.detections:
      for id, detection in enumerate(self.results.detections):
        bboxC = detection.location_data.relative_bounding_box
        ih,iw,ic = img.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
              int(bboxC.width * iw), int(bboxC.height * ih)   
        bboxs.append([bbox,detection.score])    
        if draw:
          img = self.fancyDraw(img,bbox)
          cv.putText(img,f'{int(detection.score[0] * 100)}%', 
                    (bbox[0],bbox[1]-20),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    2,(255,255,255),
                    thickness=1)
    return img,bboxs
  
  def fancyDraw(self,img,bbox, l = 30,t=4, rt=1):
      x,y,w,h = bbox
      x1,y1 = x+w,y+h
      cv.rectangle(img,bbox,(0,255,0),rt)
      # Top left x,y
      cv.line(img,(x,y), (x+l,y),
              (0,255,0),thickness= t)
      cv.line(img,(x,y), (x,y+l),
              (0,255,0),thickness= t)

      # Top Right x1,y
      cv.line(img,(x1,y), (x1-l,y),
              (0,255,0),thickness= t)
      cv.line(img,(x1,y), (x1,y+l),
              (0,255,0),thickness= t)

      # Bottom Right x1,y1
      cv.line(img,(x1,y1), (x1-l,y1),
              (0,255,0),thickness= t)
      cv.line(img,(x1,y1), (x1,y1-l),
              (0,255,0),thickness= t)

      # Bottom Left x,y1
      cv.line(img,(x,y1), (x+l,y1),
              (0,255,0),thickness= t)
      cv.line(img,(x,y1), (x,y1-l),
              (0,255,0),thickness= t)

      return img

def main():
  cap = cv.VideoCapture(0)
  pTime = 0
  detector = faceDetector()
  while True:
    success, img = cap.read()
    img,bboxs = detector.findFaces(img)
    print(bboxs)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv.putText(img,f'FPS: {int(fps)}', (0,70),cv.FONT_HERSHEY_PLAIN,2,(255,255,255),thickness=1)
    cv.imshow("Image",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv.destroyAllWindows()

if __name__ == '__main__':
    main()