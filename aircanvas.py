import cv2 # cv2 is the computer vision library
import numpy as np # numpy is the library which handles arrays
import streamlit as st # Streamlit is a Python framework used for interacting through apps over the internet
from PIL import Image # PIL stands for the Python Image Library which handles images in an effective way
import google.generativeai as genai # google.generativeai model to connect with the AI
from cvzone.HandTrackingModule import HandDetector # cvzone library which detects the hand tracking module

genai.configure(api_key="AIzaSyCXLz_dgX1ASRh42NwjQnNjU2Fd7w3D6Pg") # configuring the generative AI model with the API key
model = genai.GenerativeModel('gemini-1.5-flash') # initializing the generative AI model

prev_pos = None # assuming the previous position of the index finger is None
canvas = None # assuming the canvas doesn't contain anything
photo_combine = None # assuming the combined photo is None
output = "" # initializing the output variable as an empty string
selected_color = (255, 0, 0) # Globally selected color or the initial color is Blue
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 0, 0), (255, 255, 255)] # list of colors to draw on the air canvas

st.set_page_config(page_title="Computer Vision", layout="wide") # setting the name of the window
col1, col2 = st.columns([3, 1]) # Defining 2 columns where col1 is for the image and col2 is for the answer

with col1: # handling the first column's content
    st.title("Virtual Air Canvas Integrated With Artificial Intelligence") # setting the heading for column 1
    run = st.markdown(''':red[Below is the air canvas where the image is captured and it is sent to the]  :rainbow[AI Model]''') # displaying instructions in markdown
    fr = st.image([]) # placeholder for the image

with col2: # handling the second column's content
    st.title("Answer of the Text Written in the Canvas:") # setting the title for column 2
    op = st.markdown("The Answer for the Virtual Image is:") # output of the image will be displayed using the op variable

cap = cv2.VideoCapture(1) # capturing the video using the external camera
detector = HandDetector(detectionCon=0.2, maxHands=1) # Hand detecting function to detect the hands in the image

def getHandInfo(photo): # The function which is used to get the landmarks of the hand (21 landmarks)
    findhand, photo = detector.findHands(photo) # which finds the hands in the image
    if findhand: # if hands are detected
        hand1 = findhand[0] # extract the first hand detected
        lmList1 = hand1['lmList'] # get the list of landmarks for the first hand
        fingers1 = detector.fingersUp(hand1) # fingers1 is used to find if the fingers are up or down
        return fingers1, lmList1 # returns the landmark list of the hands and the fingers1
    else:
        return None # return None if no hands are detected

def color_bar(photo, lmList=None): # it is the function in which a color bar is displayed on the image
    global selected_color # global selected_color indicates a color which is selected globally 
    x, y = 50, 50 # positions of the color bar
    x1, y1 = 50, 300 # width and the height of the color bar
    div = y1 // len(colors) # calculating the division of the color bar based on the number of colors
    for i, color in enumerate(colors): # iterating over each color in the list
        cv2.rectangle(photo, (x, y + i * div), (x + x1, y + (i + 1) * div), color, -1) # drawing a rectangle for each color
    if lmList: # if landmarks are provided
        f1, f2 = lmList[8][:2] # get the coordinates of the index finger
        if x <= f1 < x + x1 and y <= f2 <= y + y1: # check if the index finger is within the color bar area
            color_index = (f2 - y) // div # calculate which color is selected based on finger position
            selected_color = colors[color_index] # update the globally selected color

def draw(info, prev_pos, canvas): # function to draw on the canvas based on hand movements
    fingers1, lmList1 = info # unpacking the fingers' status and landmark list
    curr_pos = None # initializing the current position as None
    if fingers1 == [0, 1, 0, 0, 0]: # if the index finger is up and others are down
        curr_pos = lmList1[8][:2] # get the current position of the index finger
        if prev_pos is None: # if there is no previous position
            prev_pos = curr_pos # set the current position as the previous position
        cv2.line(canvas, tuple(curr_pos), tuple(prev_pos), selected_color, 10) # draw a line on the canvas between the previous and current positions
    return curr_pos # return the current position

def ai(model, canvas, fingers1): # function to interact with the AI model based on hand gestures
    if fingers1 == [1, 1, 1, 0, 0]: # if the first three fingers are up
        pil_image = Image.fromarray(canvas) # convert the canvas to a PIL image
        response = model.generate_content(["Solve the mathematical problem depicted in the image below:", pil_image, "Provide a detailed explanation and solution for the problem."]) # send the image to the AI model and get the response
        return response.text # return the AI-generated text
    elif fingers1 == [1, 0, 0, 0, 0]: # if only the thumb is up
        canvas.fill(0) # clear the canvas
        return "" # return an empty string as output
    elif fingers1 == [0, 0, 0, 0, 0]: # if all fingers are down
        return "" # return an empty string as output

while True: # starting an infinite loop to continuously capture video frames
    status, photo = cap.read() # reading a frame from the video capture
    photo = cv2.flip(photo, 1) # flipping the photo horizontally for a mirror effect
    if canvas is None: # if the canvas is not initialized
        canvas = np.zeros_like(photo) # create a blank canvas with the same size as the photo
    color_bar(photo) # display the color bar on the photo
    info = getHandInfo(photo) # get hand information including finger positions and landmarks
    if info: # if hand information is detected
        fingers, lmList = info # extract the fingers status and landmarks list
        color_bar(photo, lmList) # update the color bar based on hand position
        prev_pos = draw(info, prev_pos, canvas) # draw on the canvas based on hand movement and update previous position
        output = ai(model, canvas, fingers) # interact with the AI model if a specific gesture is detected
    photo_combine = cv2.addWeighted(photo, 0.5, canvas, 0.5, 0) # blend the original photo and the canvas for display
    fr.image(photo_combine, channels="BGR", width=1000) # display the combined photo on the Streamlit interface
    op.markdown(f"<h4>{output}</h4>", unsafe_allow_html=True) # display the AI-generated output text on the Streamlit interface
