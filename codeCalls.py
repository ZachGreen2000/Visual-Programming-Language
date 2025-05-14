import mediapipe as mp
import cv2
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class codingFunctionality(): 
    def __inti__(self):
        print("temp")
    # this function handles the functionality of the print node
    def Print(output):
        return
    # this function handles the functionality of the if node
    def If():
        return
    # this function handles the functionality of the run node
    def Run():
        return 
    # this function handles the functionality of the timer node
    def Timer():
        return
    # this function handles the functionality of the loop node
    def ForLoop():
        return
    # this function handles the functionality of the input node
    def Input(input):
        return
    # this function handles the functionality of the add node
    def Add(x, y):
        return
    # this function handles the functionality of the subtract node
    def Subtract(x, y):
        return
    # this function handles the functionality of the divide node
    def Divide(x, y):
        return
    # this function handles the functionality of the equals node
    def Equals(x, y):
        return
    # this function handles the functionality of the more than node
    def MoreThan(x, y):
        return
    # this function handles the functionality of the less than node
    def LessThan(x, y):
        return
# test code    
model_path = os.path.abspath("gesture_recognizer.task")
print("Resolved model path:", model_path)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE  # Use IMAGE mode for a simple test
)

recognizer = GestureRecognizer.create_from_options(options)
print("Model loaded successfully.")