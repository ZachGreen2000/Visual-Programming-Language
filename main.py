import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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

    def get_bounds(self):
        x1 = int(self.x - 20 * self.scale)
        y1 = int(self.y + 10 * self.scale)
        x2 = int(self.x + 25 * self.scale)
        y2 = int(self.y + 40 * self.scale)
        return x1, y1, x2, y2

#this class will draw the virual keyboard used for input for the code system
class VirtualKeyboard():
    def __init__(self, img):
        # variables for positioning 
        self.offsetX = 100
        self.offsetY = 700
        self.scale = 1.5
        self.img = img
        # storage for keys
        self.keys = [
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
            ["Z", "X", "C", "V", "B", "N", "M", "SPACE"]
        ]
    # handles drawing of the keyboard
    def drawKeyboard(self):
        key_x = self.offsetX
        key_y = self.offsetY
        scale = self.scale
        img = self.img

        #loop through rows of keys
        for row in self.keys:
            #loop through keys on row
            for key in row:
                key_button = drawBase(img, key_x, key_y, scale, (220,220,210), -1, key)
                key_button.drawBase()
                # increment positioning
                key_x += 80 
            # increment row positioning
            key_x = self.offsetX
            key_y += 55
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

# this class handles the connector tool drawing
class Connector():
    def __init__(self, img):
        self.img = img
        cv2.circle(img, (120, 130), 5, (255,255,245), -1)
        cv2.line(img, (120, 130), (150, 130), (255,255,245), 2)
        cv2.circle(img, (150, 130), 5, (255,255,245), -1)
    
    # this function handles the drawing of the active connections
    def drawConnection(self):
        return

# this class draws the background for the main GUI
class DrawBackground():
    def __init__(self, img):
        self.img = img
        cv2.rectangle(img, (10, 20), (1220, 100), (19,69,139), -1) # node background
        cv2.rectangle(img, (50, 700), (870, 950), (19,69,139), -1) # keyboard background
        cv2.rectangle(img, (10, 120), (158, 140), (19,69,139), -1) # connector background
        cv2.putText(img, "Connector:", (12, 135), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0)) # connector text 

# this class handles the drawing for the output area
class DrawOutput():
    def __init__(self, img):
        self.img = img
        cv2.rectangle(img, (885, 120), (1250, 950), (19,69,139), -1)
        cv2.putText(img, "OUTPUT", (940, 170), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0)) # output title
        cv2.line(img, (895, 200), (1240, 200), (0,0,0), 2)

#this class draws a cursor at finger location
class DrawCursor():
    def __init__(self, img):
        self.img = img

# this class handles the gesture recognition
class GestureRecognizer():
    def __init__(self, img, model_path, nodes, index_finger, width, height):
        self.BaseOptions = mp.tasks.BaseOptions
        self.GestureRecognizer = vision.GestureRecognizer
        self.GestureRecognizerOptions = vision.GestureRecognizerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        options = self.GestureRecognizerOptions(base_options = self.BaseOptions(model_path),
                                               running_mode=self.VisionRunningMode.LIVE_STREAM,
                                                 result_callback=self.IdentifyGesture)
        self.recognizer = self.GestureRecognizer.create_from_options(options)
        self.dragged_node = None # detects node dragging
        self.placed_nodes = {} # storage for placed nodes
        self.is_dragging = False # detects when dragging happening

    # this function handles the gesture recognition
    def IdentifyGesture(self, result, output_image: mp.Image, timestamp_ms: int):
        if result.gestures:
            confidence = result.gestures[0][0].score
            gesture = result.gestures[0][0].cetegory_name # stores current gesture
            if confidence > 0.5 and gesture:
                # if gesture is closed fist then node is places and stored
                if gesture == "Closed_Fist":
                    self.placed_nodes.append(self.dragged_node)
                    self.dragged_node = None
                    self.is_dragging = False
    
    def draggedNodes(self, img, nodes, index_finger, width, height):
        # for index finger collision
        finger_x = int(index_finger.x * width * 2)
        finger_y = int(index_finger.y * height * 2)
        if not self.is_dragging:
            for label, box in nodes.get_boxes().items():
                x1, y1, x2, y2 = box.get_bounds()
                if x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
                    self.dragged_node = drawBase(img, finger_x, finger_y, box.scale, box.color, -1, label)
                    self.is_dragging = True
                    break
                else:
                     # for continued dragging
                    self.dragged_node.x = finger_x
                    self.dragged_node.y = finger_y
                    self.dragged_node.drawBase()
        # draw dragged nodes
        for node in self.placed_nodes:
            node.drawBase()

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
        # enabling api use
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
                    index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    # call gesture recogniser class
                    gr = GestureRecognizer(image, model_path, nodes, index_finger, width, height)
                    gr.draggedNodes()
                # gui
                flipped_image = cv2.flip(image, 1)
                width = int(flipped_image.shape[1] * 2)
                height = int(flipped_image.shape[0] * 2)
                resized_image = cv2.resize(flipped_image, (width, height))
                DrawBackground(resized_image)
                DrawOutput(resized_image)
                Connector(resized_image)
                nodes = GUINodes(resized_image)
                vk = VirtualKeyboard(resized_image)
                vk.drawKeyboard()
                # below code displays image and gives ability to end stream on q press
                cv2.imshow('MediaPipe Hands', resized_image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        self.cap.release()
        cv2.destroyAllWindows() 
model_path = "gesture_recognizer.task" # sets path to gesture recogniser api
app = mainLoop(model_path, 0)
app.run()