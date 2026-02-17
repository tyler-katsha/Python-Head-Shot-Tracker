import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector():
  def __init__(self,staticMode = False,
               maxFaces = 2,refine_landmarks=False, minDetectionCon = 0.5,
               minTrackCon = 0.5):
    self.staticMode = staticMode
    self.maxFaces = maxFaces
    self.refine_landmarks = refine_landmarks
    self.minDetectionCon = minDetectionCon
    self.minTrackCon = minTrackCon
    self.mpDraw = mp.solutions.drawing_utils
    self.mpFaceMesh = mp.solutions.face_mesh
    self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.refine_landmarks,
                                             self.minDetectionCon,self.minTrackCon)
    self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius = 1)
    self.faces = []

  def findFaceMesh(self,img,draw = True):
    self.imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    self.results = self.faceMesh.process(self.imgRGB)
    self.faces = []
    if self.results.multi_face_landmarks:
      for faceLms in self.results.multi_face_landmarks:
        if draw:
          self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_TESSELATION,
                                    self.drawSpec,self.drawSpec)
        # By the connections you can use different mesh functions such as 
        # FACEMESH_TESSELATION -> full mesh
        # FACEMESH_CONTOURS -> outlin
        # FACEMESH_IRISES -> Iris
        face = []
        
        for id,lm in enumerate(faceLms.landmark):
          #print(lm)
          ih,iw,ic = img.shape
          x,y= int(lm.x * iw) , int(lm.y * ih)
          #cv.putText(img, str(id),(x,y),cv.FONT_HERSHEY_PLAIN,
              #0.8,(0,255,0),1)
          #print(id,x,y)
        
          face.append([id,x,y])
        self.faces.append(face)
    return img,self.faces

def main():
  cap = cv.VideoCapture(0)
  pTime = 0
  detector = FaceMeshDetector()
  while True:
    success,img = cap.read()
    img, faces = detector.FindFaceMesh(img,False)
    #print(faces)
    if len(faces) != 0:
      print(f'Faces: {len(faces)}')
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'Fps: {int(fps)}',(20,70),cv.FONT_HERSHEY_PLAIN,
              2,(0,255,0),2)
    cv.imshow('image',img)
    if cv.waitKey(1) & 0xFF in [ord('q'),ord('Q')]:
      break

  cap.release()
  cv.destroyAllWindows()


if __name__ == '__main__':
  main()