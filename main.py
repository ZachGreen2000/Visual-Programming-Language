import mediapipe as mp
import cv2
import codeCalls

# this classdraws the shape of each block and adds the text
class drawBase():
    def __init__(self, img, x, y, scale, color, thickness, text):
        self.img = img
        self.x = x
        self.y = y
        self.scale = scale
        self.color = color
        self.thickness = thickness
        self.text = text

    def drawBase(self):
        cv2.rectangle(self.img, (self.x - int(10 * self.scale), self.y + int(20 * self.scale)), (self.x + int(10 * self.scale), self.y + int(60 * self.scale)), self.color, self.thickness)
        cv2.putText(self.img, self.text, (self.x + int(10 * self.scale), self.y + int(20 * self.scale)), cv2.FONT_HERSHEY_COMPLEX, 0.8, self.color)

# this class is for drawing the gui for the program using a data store for each block
class GUINodes():
    def __init__(self, img):
        # dictionairy for each node, draws them on screen and stores them for referenceing 
        boxes = {
            "Print": drawBase(img, 20, 40, 1, (0,0,255), -1, "Print"),
            "If": drawBase(img, 80, 90, 1, (0,0,255), -1, "If Statement"),
            "Run": drawBase(img, 100, 150, 1, (255,0,0), -1, "Run"),
            "Timer": drawBase(img, 180, 200, 1, (0,255,0), -1, "Timer"),
            "For": drawBase(img, 200, 250, 1, (255,255,0), -1, "For Loop"),
            "Input": drawBase(img, 250, 300, 1, (255,255,245), -1, "Input"),
            "Add": drawBase(img, 50, 20, 1, (120,120,110), -1, "Add"),
            "Subtract": drawBase(img, 150, 120, 1, (220,220,210), -1, "Subtract"),
            "Divide": drawBase(img, 150, 120, 1, (220,220,210), -1, "Divide"),
            "Equals": drawBase(img, 150, 120, 1, (220,220,210), -1, "Equals"),
            "LessThan": drawBase(img, 150, 120, 1, (220,220,210), -1, "Less Than"),
            "MoreThan": drawBase(img, 150, 120, 1, (220,220,210), -1, "More Than")
        }
        # handles the actual drawing of nodes
        for box in boxes:
            box.drawBase()

# this class houses the logic for the main running loop of the hand tracking
class mainLoop():
    def __init__(self, model_path, webcam_id): # init function sets variables for use
        self.cap = cv2.VideoCapture(webcam_id) ## sets video capture with webcam id 0
        if not self.cap.isOpened():
            print("error openning webcam") # debug
            return
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.model_path = model_path

    # this function is for the moving of nodes when they are selected by user
    def MoveNodes():
        return
    
    # this function is to confirm the placement of nodes on screen
    def PlaceNodes():
        return
    
    # this function is to draw the lines that connect the nodes
    def ConnectNodes():
        return
    
    # this function is for drawing the main background ui and other things
    def MainGUI():
        return
    
    # this function is called to run the app and detects hand and draws landmarks
    # using the landmarks and gesture recognition it calls functions for gesture based coding implementation
    def run(self):
        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened(): # main while loop to run while cam is open, store all logic insise
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.") # debug
                    continue
                if image is None or image.size == 0:
                    print("Error: Empty frame") # debug
                    break
                
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks: # this detects hands and draws landmarks on hands
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                
                # gui
                flipped_image = cv2.flip(image, 1)
                GUINodes(flipped_image)
                # below code displays image and gives ability to end stream on q press
                cv2.imshow('MediaPipe Hands', cv2.flip(flipped_image, 1))
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        self.cap.release()
        cv2.destroyAllWindows() 
model_path = "gesture_recognizer.task" # sets path to gesture recogniser api
app = mainLoop(model_path, 0)
app.run()