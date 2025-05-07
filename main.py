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
        cv2.rectangle(self.img, (self.x - int(20 * self.scale), self.y + int(10 * self.scale)), (self.x + int(25 * self.scale), self.y + int(40 * self.scale)), self.color, self.thickness)
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_COMPLEX, 0.2 * self.scale, 1)[0] # get text size
        text_x = self.x - int(text_size[0] / 2) + 2 # adjust text position to centre
        text_y = self.y + int(text_size[1] / 2) + 45 # same as above
        cv2.putText(self.img, self.text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.2 * self.scale, (0,0,0))

#this class will draw the virual keyboard used for input for the code system
class VirtualKeyboard():
    def __init__(self, img):
        # variables for positioning 
        self.offsetX = 50
        self.offsetY = 400
        self.scale = 1.5
        self.img = img
        # storage for keys
        self.keys = [
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
            ["Z", "X", "C", "V", "B", "N", "M"]
        ]

    # handles drawing of the keyboard
    def drawKeyboard(self, img, offsetX, offsetY, scale):
        key_x = offsetX
        key_y = offsetY

        #loop through rows of keys
        for row in self.keys:
            #loop through keys on row
            for key in row:
                key_button = drawBase(img, key_x, key_y, scale, (255, 255, 255), 2, key)
                key_button.drawBase()
                # increment positioning
                key_x += 50 * scale
            # increment row positioning
            key_x = offsetX
            key_y += 40 * scale
    # for key storage
    def get_keys(self):
        return self.keys

# this class is for drawing the gui for the program using a data store for each block
class GUINodes():
    def __init__(self, img):
        # horizontal gap between nodes
        offsetX = 100
        # vertical gap between nodes
        offsetY = 10
        #scale
        scale = 2
        # dictionairy for each node, draws them on screen and stores them for referenceing 
        self.boxes = {
            "Print": drawBase(img, 60, offsetY, scale, (0,0,255), -1, "Print"),
            "If": drawBase(img, 60 + offsetX, offsetY, scale, (0,0,255), -1, "If"),
            "Run": drawBase(img, 60 + 2 * offsetX, offsetY, scale, (255,0,0), -1, "Run"),
            "Timer": drawBase(img, 60 + 3 * offsetX, offsetY, scale, (0,255,0), -1, "Timer"),
            "For": drawBase(img, 60 + 4 * offsetX, offsetY, scale, (255,255,0), -1, "For"),
            "Input": drawBase(img, 60 + 5 * offsetX, offsetY, scale, (255,255,245), -1, "Input"),
            "Add": drawBase(img, 60 + 6 * offsetX, offsetY, scale, (120,120,110), -1, "Add"),
            "Subtract": drawBase(img, 60 + 7 * offsetX, offsetY, scale, (220,220,210), -1, "Subtract"),
            "Divide": drawBase(img, 60 + 8 * offsetX, offsetY, scale, (220,220,210), -1, "Divide"),
            "Equals": drawBase(img, 60 + 9 * offsetX, offsetY, scale, (220,220,210), -1, "Equals"),
            "LessThan": drawBase(img, 60 + 10 * offsetX, offsetY, scale, (220,220,210), -1, "Less Than"),
            "MoreThan": drawBase(img, 60 + 11 * offsetX, offsetY, scale, (220,220,210), -1, "More Than")
        }
        # handles the actual drawing of nodes
        for box in self.boxes.values():
            box.drawBase()
    # for storage of disctionairy
    def get_boxes(self):
        return self.boxes

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
                width = int(flipped_image.shape[1] * 2)
                height = int(flipped_image.shape[0] * 2)
                resized_image = cv2.resize(flipped_image, (width, height))
                GUINodes(resized_image)
                VirtualKeyboard(resized_image)
                # below code displays image and gives ability to end stream on q press
                cv2.imshow('MediaPipe Hands', resized_image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        self.cap.release()
        cv2.destroyAllWindows() 
model_path = "gesture_recognizer.task" # sets path to gesture recogniser api
app = mainLoop(model_path, 0)
app.run()