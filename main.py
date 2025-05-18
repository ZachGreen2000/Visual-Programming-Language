import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import os

# this classdraws the shape of each block and adds the text
class drawBase():
    def __init__(self, x, y, scale, color, thickness, text):
        #self.img = img
        self.x = x
        self.y = y
        self.scale = scale
        self.color = color
        self.thickness = thickness
        self.text = text

    def drawBase(self, img):
        cv2.rectangle(img, (self.x - int(20 * self.scale), self.y + int(10 * self.scale)), (self.x + int(25 * self.scale), self.y + int(40 * self.scale)), self.color, self.thickness)
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_COMPLEX, 0.2 * self.scale, 1)[0] # get text size
        text_x = self.x - int(text_size[0] / 2) + 2 # adjust text position to centre
        text_y = self.y + int(text_size[1] / 2) + 45 # same as above
        cv2.putText(img, self.text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.2 * self.scale, (0,0,0))

    def get_bounds(self):
        x1 = int(self.x - 20 * self.scale)
        y1 = int(self.y + 10 * self.scale)
        x2 = int(self.x + 25 * self.scale)
        y2 = int(self.y + 40 * self.scale)
        return x1, y1, x2, y2

#this class will draw the virual keyboard used for input for the code system
class VirtualKeyboard():
    def __init__(self):
        # variables for positioning 
        self.offsetX = 890
        self.offsetY = 125
        self.scale = 1.5
        self.img = None
        # storage for keys
        self.keys = [
            ["1", "2", "3", "4", "5"],
            ["6", "7", "8", "9", "0"],
            ["Q", "W", "E", "R", "T"],
            ["Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G"],
            ["H", "J", "K", "L","Z"],
            ["X", "C", "V", "B", "N"],
            ["M", "SPACE", "ENTER"]
        ]
        self.input_text = ""
        self.keyLocations = {}
        self.last_press_time = 0
        self.inputStore = []
        self.gr = None
    # handles drawing of the keyboard
    def drawKeyboard(self, img):
        key_x = self.offsetX
        key_y = self.offsetY
        scale = self.scale
        self.img = img

        #loop through rows of keys
        for row in self.keys:
            #loop through keys on row
            for key in row:
                key_button = drawBase(key_x, key_y, scale, (220,220,210), -1, key)
                key_button.drawBase(img)
                x1, y1, x2, y2 = key_button.get_bounds()
                self.keyLocations[key] = (x1, y1, x2, y2)
                # increment positioning
                key_x += 80 
            # increment row positioning
            key_x = self.offsetX
            key_y += 55
    # for key storage
    def get_keys(self):
        return self.keys
    
    # this function is for keyboard collision but only when certain nodes are placed
    def proccessKeyboard(self, activeNode, index_finger, width, height):
        if activeNode != "Input":
            return
        
        current_time = time.time()
        if current_time - self.last_press_time < 1.0: 
            return # return if too soon since last key press
        print("Active node is: ", activeNode) # debug
        # for index finger collision
        finger_x = int((1 - index_finger.x) * width)
        finger_y = int(index_finger.y * height)

        cv2.circle(self.img, (finger_x, finger_y), 10, (10,200,235), -1)

        # to loop through keys
        for key, (x1, y1, x2, y2) in self.keyLocations.items():
            if x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
                if key == "ENTER": # check for enter press to finish the keyboard string
                    print("Enter pressed") # debug
                    if self.input_text not in self.inputStore:
                        self.inputStore.append(self.input_text)
                    self.input_text = ""
                    activeNode = ""
                    self.gr.setActiveNode(activeNode)
                    return
                else:
                    self.input_text += " " if key == "SPACE" else key # get key collision
                self.last_press_time = current_time
                print(self.input_text)
                return
    # this script gets correct gesture class instance
    def getScriptInstance(self, gestureIdentifier):
        self.gr = gestureIdentifier
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
            "Print": drawBase(60, offsetY, scale, (0,0,255), -1, "Print"),
            "If": drawBase(60 + offsetX, offsetY, scale, (0,100,255), -1, "If"),
            "Run": drawBase(60 + 2 * offsetX, offsetY, scale, (255,0,0), -1, "Run"),
            "Timer": drawBase(60 + 3 * offsetX, offsetY, scale, (0,255,0), -1, "Timer"),
            "For": drawBase(60 + 4 * offsetX, offsetY, scale, (255,255,0), -1, "For"),
            "Input": drawBase(60 + 5 * offsetX, offsetY, scale, (255,255,245), -1, "Input"),
            "Add": drawBase(60 + 6 * offsetX, offsetY, scale, (120,120,110), -1, "Add"),
            "Subtract": drawBase(60 + 7 * offsetX, offsetY, scale, (220,220,210), -1, "Subtract"),
            "Divide": drawBase(60 + 8 * offsetX, offsetY, scale, (50,220,110), -1, "Divide"),
            "Equals": drawBase(60 + 9 * offsetX, offsetY, scale, (220,89,110), -1, "Equals"),
            "LessThan": drawBase(60 + 10 * offsetX, offsetY, scale, (200,150,210), -1, "Less Than"),
            "MoreThan": drawBase(60 + 11 * offsetX, offsetY, scale, (90,220,200), -1, "More Than")
        }
        # handles the actual drawing of nodes
        for box in self.boxes.values():
            box.drawBase(img)
    # for storage of disctionairy
    def get_boxes(self):
        return self.boxes

# this class handles the connector tool drawing
class Connector():
    def __init__(self):
        self.img = None
        self.connect_Mode = False
        #print("connect mode is", self.connect_Mode) # debug

    # this function draws the connector tool on background
    def drawTool(self, img):
        self.img = img
        cv2.circle(img, (120, 130), 5, (255,255,245), -1)
        cv2.line(img, (120, 130), (150, 130), (255,255,245), 2)
        cv2.circle(img, (150, 130), 5, (255,255,245), -1)

    # this function detects connector tool collision
    def detectConnection(self, index_finger, width, height):
        x1 = 120 
        y1 = 130
        x2 = 150
        y2 = 130 # same as tool location
        # for index finger collision
        self.finger_x = int((1 - index_finger.x) * width)
        self.finger_y = int(index_finger.y * height)
        if not self.connect_Mode:
            if x1 <= self.finger_x <= x2 and y1 <= self.finger_y <= y2: # detect collision
                self.connect_Mode = True
                #print("connector tool collision at") #debug
        # runs when connect mode on to draw tool at finger tip
        if self.connect_Mode:
            cv2.circle(self.img, (self.finger_x, self.finger_y), 10, (255,255,255), -1)
            #print("connect mode is", self.connect_Mode) # debug
    
    # this function calls on open palm to stop using connect tool
    def stopConnectTool(self):
        self.connect_Mode = False
        #print("self connect mode is", self.connect_Mode) # debug

    # this gets connect mode
    def getConnectMode(self):
        return self.connect_Mode

# this class draws the background for the main GUI
class DrawBackground():
    def __init__(self, img):
        self.img = img
        cv2.rectangle(img, (10, 20), (1220, 100), (19,69,139), -1) # node background
        cv2.rectangle(img, (845, 120), (1260, 600), (19,69,139), -1) # keyboard background
        cv2.rectangle(img, (10, 120), (158, 140), (19,69,139), -1) # connector background
        cv2.putText(img, "Connector:", (12, 135), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0)) # connector text 

# this class handles the drawing for the output area
class DrawOutput():
    def __init__(self, img):
        self.img = img
        cv2.rectangle(img, (50, 650), (1250, 950), (19,69,139), -1) # background
        cv2.putText(img, "OUTPUT", (100, 710), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0)) # output title
        cv2.putText(img, "INPUT", (900, 710), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0))
        cv2.line(img, (55, 720), (1245, 720), (0,0,0), 2)
        cv2.line(img, (845, 660), (845, 940), (0,0,0), 2)

#this class draws a cursor at finger location
class DrawCursor():
    def __init__(self, img):
        self.img = img

# this class handles the gesture recognition
class GestureRecognizer():
    def __init__(self, model_path):
        self.BaseOptions = mp.tasks.BaseOptions
        self.GestureRecognizer = vision.GestureRecognizer
        self.GestureRecognizerOptions = vision.GestureRecognizerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        options = self.GestureRecognizerOptions(base_options = self.BaseOptions(model_asset_path=model_path),
                                               running_mode=self.VisionRunningMode.LIVE_STREAM,
                                                 result_callback=self.IdentifyGesture)
        self.recognizer = self.GestureRecognizer.create_from_options(options)
        self.dragged_node = None # detects node dragging
        self.placed_nodes = {} # storage for placed nodes
        self.is_dragging = False # detects when dragging happening
        # delay tracking variables
        self.hovered_node = None
        self.hover_start_time = None
        self.drag_delay = 0.5
        self.con = None
        # process connections variables
        self.connection_start_time = None
        self.connections = []
        self.img = None
        self.activeNode = ""
        self.VK = None
        self.result = None
        self.functionRunning = False
        self.notRun = True

    # this function handles the gesture recognition
    def IdentifyGesture(self, result, output_image: mp.Image, timestamp_ms: int):
        #img = self.img.get(timestamp_ms)
        if result.gestures:
            confidence = result.gestures[0][0].score
            gesture = result.gestures[0][0].category_name # stores current gesture
            if confidence > 0.5 and gesture:
                #print(gesture) #debug
                # if gesture is closed fist then node is places and stored
                if gesture == "Closed_Fist" and self.dragged_node is not None:
                    if self.dragged_node['text'] == "Input" and "Input" in self.placed_nodes: # takes into account a second input node
                        self.placed_nodes['Input2'] = drawBase(self.dragged_node['x'],
                                                                            self.dragged_node['y'],
                                                                            self.dragged_node['scale'],
                                                                            self.dragged_node['color'],
                                                                            self.dragged_node.get('thickness', -1),
                                                                            self.dragged_node['text'])
                    else:
                        self.placed_nodes[self.dragged_node['text']] = drawBase(self.dragged_node['x'],
                                                                                self.dragged_node['y'],
                                                                                self.dragged_node['scale'],
                                                                                self.dragged_node['color'],
                                                                                self.dragged_node.get('thickness', -1),
                                                                                self.dragged_node['text'])
                    self.activeNode = self.dragged_node['text']
                    self.dragged_node = None
                    self.is_dragging = False
                    self.hovered_node = None
                    self.hover_start_time = None
                if gesture == "Open_Palm":
                    self.con.stopConnectTool() # stop connect tool on open palm gesture
                    #print("connect mode stopped") #debug
                if gesture == "Thumb_Down": # resets screen for user to start again
                    # clear all stores
                    self.placed_nodes.clear()
                    self.connections.clear()
                    self.VK.inputStore.clear()
                    self.VK.input_text = ""
                    self.activeNode = ""
                    self.functionRunning = False
                    print("Refreshed for starting again")
    # gets script instances
    def setScriptInstance(self, connector, VK):
        self.con = connector
        self.VK = VK
    # gets active node
    def setActiveNode(self, node):
        self.activeNode = node
    # this function handles the dragging of nodes into the workspace
    def draggedNodes(self, img, nodes, index_finger, width, height, timestamp_ms):
        self.img = img
        # for index finger collision
        finger_x = int((1 - index_finger.x) * width)
        finger_y = int(index_finger.y * height)

        if not self.is_dragging:
            hovering_now = False
            for label, box in nodes.get_boxes().items():
                x1, y1, x2, y2 = box.get_bounds()
                if x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
                    #print("finger colliding with: " + label) #debug
                    hovering_now = True
                    if self.hovered_node == label:
                        #print("hovered node is equal to label") #debug
                        if time.time() - self.hover_start_time >= self.drag_delay: # check if finger hovering for long enough
                            #print("Drawing node at: " + self.hovered_node, finger_x, finger_y) #debug
                            self.dragged_node = {'x': finger_x, 'y': finger_y, 'scale': box.scale, 'color': box.color, 'thickness': -1, 'text': label}
                            self.is_dragging = True
                            self.hovered_node = None
                            self.hover_start_time = None
                    else:
                        self.hovered_node = label
                        self.hover_start_time = time.time()
                    break
            if not hovering_now: # reset finger if not hovering on node
                self.hovered_node = None
                self.hover_start_time = None
        else:
            # for continued dragging
            self.dragged_node['x'] = finger_x
            self.dragged_node['y'] = finger_y
            DN = drawBase(**self.dragged_node)
            DN.drawBase(img)
            cv2.circle(img, (finger_x, finger_y), 10, (255,255,255), -1)
            #print("node is now dragging: ", self.dragged_node.x, self.dragged_node.y) #debug
            #print("node scale and color is: ", self.dragged_node.scale, self.dragged_node.color) #debug
        # draw dragged nodes
        for node in self.placed_nodes.values():
            node.drawBase(img)

    # this function will handle the connections made between places nodes and store the connections for use
    def process_connections(self, img, index_finger, width, height):
        # for index finger collision
        finger_x = int((1 - index_finger.x) * width)
        finger_y = int(index_finger.y * height)

        connect_mode = self.con.getConnectMode() # get connecting mode result
        if connect_mode:
            for label, node in self.placed_nodes.items():
                x1, y1, x2, y2 = node.get_bounds()
                if x1 <= finger_x <= x2 and y1 <= finger_y <= y2: # collision
                    if self.connection_start_time is not None and time.time() - self.connection_start_time >= 1.0:
                        print("finger colliding with: ", label)
                        node_text = label
                        if node_text not in self.connections:
                            self.connections.append(node_text)
                        self.connection_start_time = None
                        break
                    elif self.connection_start_time is None: # logic for time delay
                        self.connection_start_time = time.time()
                        break

            # for drawing from last node to finger tip
            if self.connections:
                last_label = self.connections[-1]
                last_node = self.placed_nodes[last_label]
                x1, y1 = int(last_node.x), int(last_node.y)
                cv2.line(img, (x1,y1), (finger_x, finger_y), (255,255,255), 2)

        # for line drawing between placed nodes
        if len(self.connections) > 1:
            for i  in range(len(self.connections) - 1):
                label_a = self.connections[i]
                label_b = self.connections[i + 1]

                if label_a in self.placed_nodes and label_b in self.placed_nodes:
                    node_a = self.placed_nodes[label_a]
                    node_b = self.placed_nodes[label_b]
                    x1,y1 = int(node_a.x), int(node_a.y)
                    x2,y2 = int(node_b.x), int(node_b.y)
                    cv2.line(img, (x1, y1), (x2, y2), (0,0,0), 2)
        self.VK.proccessKeyboard(self.activeNode, index_finger, width, height)
        self.executeOutput()

    # this function controls if run is activated to then activate the output of the code
    def executeOutput(self):
        if "Run" in self.connections and self.notRun:
            if self.connections[-1] == "Run":
                #print("Running output") # debug
                # set variables for use
                self.result = None
                inputs = self.VK.inputStore
                
                i = 0 # while loop for iterating through connected nodes
                while i < len(self.connections):
                    node = self.connections[i]

                    if node == "Input": # get value for input node
                        if inputs:
                            self.result = inputs.pop(0)
                            print(self.result)
                    # handles print node to show an input if there is one
                    elif node == "Print" and self.result is not None:
                        print(self.result)
                        self.functionRunning = True
                    # handles addition where if there are two input nodes it tries to add them given they can be passed to floats
                    elif node == "Add":
                        if i + 2 < len(self.connections) and self.connections[i+1] == "Input" and self.connections[i+2] == "Input2":
                            a, b = inputs.pop(0), inputs.pop(0)
                            try:
                                self.result = str(float(a) + float(b))
                                print(self.result)
                            except:
                                print("Error")
                            i += 2
                    # handles subtraction the same way as addition
                    elif node == "Subtract":
                        if i + 2 < len(self.connections) and self.connections[i+1] == "Input" and self.connections[i+2] == "Input2":
                            a, b = inputs.pop(0), inputs.pop(0)
                            try:
                                self.result = str(float(a) - float(b))
                                print(self.result)
                            except:
                                print("Error")
                            i += 2
                    # handles division the same as above with the addition of handling division by zero    
                    elif node == "Divide":
                        if i + 2 < len(self.connections) and self.connections[i+1] == "Input" and self.connections[i+2] == "Input2":
                            a, b = inputs.pop(0), inputs.pop(0)
                            try:
                                self.result = str(float(a) / float(b))
                                print(self.result)
                            except ZeroDivisionError:
                                print("Error dividing by zero")
                            except:
                                print("Error")
                            i += 2  
                    # same process for equals comparing two values to return a boolean
                    elif node == "Equals":
                        if i + 2 < len(self.connections) and self.connections[i+1] == "Input" and self.connections[i+2] == "Input2":
                            a, b = inputs.pop(0), inputs.pop(0)
                            self.result = str(a) == str(b)
                            print(self.result)

                    elif node == "LessThan":
                          if i + 2 < len(self.connections) and self.connections[i+1] == "Input" and self.connections[i+2] == "Input2":
                            a, b = inputs.pop(0), inputs.pop(0)
                            try:
                                self.result = str(float(a) < float(b))
                                print(self.result)
                            except:
                                print("Error")

                    elif node == "MoreThan":
                          if i + 2 < len(self.connections) and self.connections[i+1] == "Input" and self.connections[i+2] == "Input2":
                            a, b = inputs.pop(0), inputs.pop(0)
                            try:
                                self.result = str(float(a) > float(b))
                                print(self.result)
                            except:
                                print("Error")
                    # handles if statement node by seeing if condition is true or false and either continuing or skipping
                    elif node == "If":
                        condition = self.result
                        if condition:
                            print("continuing")
                        else:
                            print("if not true, skipping next node")
                            i += 1
                    # handles for loop, looping based on input amount for range
                    elif node == "For":
                        if inputs:
                            try:
                                count = int(inputs.pop(0))
                                for n in range(count):
                                    print(n+1)
                            except:
                                print("Error")
                    # handles a basic timer node for counting upwards based on limit set by input
                    elif node == "Timer":
                        if inputs:
                            try:
                                t = 0
                                timer = float(inputs.pop(0))
                                while t < timer:
                                    print(t)
                                    t += 1
                            except:
                                print("Error")
                    
                    i += 1 # increase for iteration
            
            else: # runs if run in incorrect place
                print("Error")

            self.notRun = True # stop running of function more than once

    # this function displays output of current node system
    def displayOutput(self, index_finger, width, height):
        if self.activeNode == "Input":
            # for index finger collision
            finger_x = int((1 - index_finger.x) * width)
            finger_y = int(index_finger.y * height)
            cv2.circle(self.img, (finger_x, finger_y), 10, (10,200,235), -1)
        # this is for displaying the current input for user
        cv2.putText(self.img, self.VK.input_text, (950, 850), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0))
        if self.functionRunning:
            print("Function is running")
            cv2.putText(self.img, self.result, (100, 850), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0)) # display in output
        # this section is for displaying the stored inputs
        if self.VK.inputStore:
            text_offset = 890
            for i in self.VK.inputStore:
                cv2.putText(self.img, i, (text_offset, 800), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0))
                text_offset += 100
        else:
            text_offset = 890

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
        self.gr = GestureRecognizer(self.model_path)
        self.con = Connector()
        self.index_finger = type('Point', (), {'x': 0, 'y': 0})() # initialise index_finger with same type for global use
        self.vk = VirtualKeyboard()

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
                
                width = int(image.shape[1] * 2)
                height = int(image.shape[0] * 2)

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                self.gr.setScriptInstance(self.con, self.vk) # inject correct connector instance into class
                self.vk.getScriptInstance(self.gr) # gived virtual keyboard class gesture class
                #get frame and call function
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                self.gr.recognizer.recognize_async(mp_image, timestamp_ms=cv2.getTickCount())
                
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
                    self.index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                #flip and resize screen    
                flipped_image = cv2.flip(image, 1)
                resized_image = cv2.resize(flipped_image, (width, height))
                # gui
                DrawBackground(resized_image)
                DrawOutput(resized_image)
                nodes = GUINodes(resized_image)
                self.vk.drawKeyboard(resized_image)
                # draws connection and detection of tool use
                self.con.drawTool(resized_image)
                
                # call function
                self.gr.draggedNodes(resized_image, nodes, self.index_finger, width, height, timestamp_ms=cv2.getTickCount())
                self.con.detectConnection(self.index_finger, width, height)
                self.gr.process_connections(resized_image, self.index_finger, width, height)
                self.gr.displayOutput(self.index_finger, width, height)

                # below code displays image and gives ability to end stream on q press
                cv2.imshow('MediaPipe Hands', resized_image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        self.cap.release()
        cv2.destroyAllWindows() 
model_path = os.path.abspath("gesture_recognizer.task") # sets path to gesture recogniser api
app = mainLoop(model_path, 0)
app.run()