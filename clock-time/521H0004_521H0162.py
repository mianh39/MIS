# Description: This program is used to extract the time from the clock image.
# Import the necessary libraries
import os
import cv2
import math
import numpy as np

#This function is used to pre-process the image
def pre_process(img):
    # Get the height and width of the image
    height, width, _ = img.shape 
    # Calculate the scale factor to resize the image to a maximum of 1000 pixels
    scale = 1000 / max(height, width)
    # Resize the image using the calculated scale factor
    img = cv2.resize(img, (int(width * scale), int(height * scale)))
    return img

#This function detects the clock by circle size or rectangle size.
def clock_detector(img, blurred):
    # Initialize variables to store the center and radius of the circle
    radius = 0
    cent_x, cent_y = 0, 0
    # Use the Hough method to find circles in the image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 400, param1=50, param2=100, minRadius=100, maxRadius=500)
    # Initialize variable to store the largest circle
    bigest_circle = None
    # Find the largest circle in the image
    if circles is not None:
        # Browse through the circles found in the image to find the largest circle
        for circle in circles[0, :]:
            # Get the radius of the circle
            x, y, r = circle
            # Update the bigger circle if needed
            if r > radius:
                bigest_circle = circle
        # Get the coordinates and radius of the largest circle
        x, y, r = bigest_circle
        cent_x = int(x)
        cent_y = int(y)
        radius = int(r)
        
    # Incase the radius is not found a circle, let find by rectangle.
    else:
        # Find objects in the image using the findContours method
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize variables to store the area and largest rectangle
        max_area = 0
        biggest_rectangle = None
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            # Update the bigger rectangle if needed
            if area > max_area:
                max_area = area
                biggest_rectangle = contour     
        if biggest_rectangle is not None:
            # Get the coordinates of the largest rectangle
            (x, y, w, h) = cv2.boundingRect(biggest_rectangle)

            # Calculate the coordinates of the center of the rectangle
            cent_x = x + w // 2
            cent_y = y + h // 2
            # Calculate the radius of the circle inscribed in the rectangle
            radius = min(w, h) // 2
    return cent_x, cent_y, radius

#This function detects the lines in the clock image
def line_detector(img, blurred):
    # Use Canny filter to find edges in image
    edges = cv2.Canny(blurred, 50, 150)
    # Use the Hough method to find lines in the edge image
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90, minLineLength=30, maxLineGap=5)
    return lines

# This function groups lines that are close together and nearly parallel to each other
def group_lines(lines, center_x, center_y, radius):
    # Initialize a list to store groups of lines
    groups =[]
    # Iterate through the lines
    for line in lines:
        # Get the coordinates of the two endpoints of the line
        x1, y1, x2, y2 = line[0]

        # Find the length of the line from the center of the clock
        length1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
        length2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)

        #Find the farthest and closest points from the center of the clock
        farthest_length = np.max([length1, length2])
        closest_length = np.min ([length1, length2])

        # The farthest point must be within the radius of the clock and the nearest point must only be within 50% of the radius of the clock
        if ((farthest_length < radius) and (closest_length < radius*50/100)):
            # Calculate the angle of the line in degrees
            angle = math.atan2(y2 - y1, x2 - x1)
            angle = math.degrees(angle)

            # Initialize flag variable to check whether the line belongs to any group or not
            grouped = False
            for group in groups:
                # Get the average angle of the group
                mean_angle = group['mean_angle']                
                # Incase the angle of the line is close to the average angle of the group or the average angle of the group plus 180 degrees
                if abs(angle - mean_angle) < 12 or abs(angle - mean_angle - 180) < 12 or abs(angle - mean_angle + 180) < 12:
                    # Add lines to the group
                    group['lines'].append(line)
                    # Set the flag variable to True to signal that the group has been found
                    grouped = True
                    break
            # If you cannot find a suitable group
            if not grouped:
                # Create a new group with its lines and angles
                groups.append({'lines': [line], 'mean_angle': angle})
    return groups

# This function to calculate the distance between two parallel lines.
def distance_lines(line_1, line_2):
    # Get the coordinates
    x1_1, y1_1, x2_1, y2_1 = line_1[0]
    x1_2, y1_2, _ , _ = line_2[0]
    # Create vector line
    vector = np.array([x2_1 - x1_1, y2_1 - y1_1])
    # Create a vector connecting a point on the other line
    vector_connect = np.array([x1_2 - x1_1, y1_2 - y1_1])
    # Distance between the two lines.
    distance_line = np.abs(np.cross(vector, vector_connect)) / np.linalg.norm(vector)
    return distance_line

# This function has the purpose of finding the farthest endpoint from the clock center of a line segment among line segments
def hands_detector(groups, center_x, center_y):
    # Initialize a list clock hands
    hands_list = []
    # Interate through the groups
    for group in groups:
        # Get the list of lines in the group
        lines = group['lines']
        total_line = len(lines)
        max_t = 0
        farthest = 0
        # Initialize the maximum length of the clock hand
        for i in range(total_line):
            x1, y1, x2, y2 = lines[i][0]
            length1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            length2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
            length = np.max([length1, length2])
            # Find the farthest point from the center of the clock
            if length > farthest:
                farthest = length
                if length == length1:
                    max_line = x1, y1, center_x, center_y
                else:
                    max_line = x2, y2, center_x, center_y
            # Interate through the other lines and calculate the distance between them
            
            j = i+1
            while(j<total_line):
                # Find the distance between two lines
                thickness = distance_lines(lines[i], lines[j])
                # Update maximum thickness
                if (thickness > max_t):
                    max_t = thickness
                j += 1
        # Create line with maximum thickness and the farthest point from the center of the clock
        line = max_line, max_t, farthest
        # If the thickness is greater than 0, it means there are at least two parallel lines
        if max_t > 0:
            # Add this set to the clock hands list
            hands_list.append(line)
    # Sort descending the list of clock hands by thickness
    hands_list.sort(key=lambda z: z[2], reverse=True)
    # The three lines with the largest thickness are hours, minutes, and seconds
    hands_list = hands_list[:3]
    return hands_list

#This function determines the hour hand, minute hand, and second hand
def determine_hands(hands_list):
    # Determine the hour hand, minute hand, and second hand based on the thickness and length of the clock hands
    hands_bythickness = sorted(hands_list, key=lambda hands_list: hands_list[1])
    # Get the second hand from the list containing 3 clock hands
    second_hand = hands_bythickness[0]       #The second hand is the thinnest hand
    hands_list.remove(second_hand)           # Remove the second hand from the list of clock hands
    # Arrange the remaining 2 clock hands by length
    hands_bylength = sorted(hands_list, key=lambda hands_list: hands_list[2])
    # Get the hour hand and minute hand from the list containing 2 clock hands
    hour_hand = hands_bylength[0]            # The hour hand is the shorter hand
    minute_hand = hands_bylength[1]          # The minute hand is the longer hand
    return hour_hand, minute_hand, second_hand


# This function draws a frame around and labels the clock hands back on the image
def draw_hands(img, hour_hand, minute_hand, second_hand):
    # Draw line and add label for hour hand
    x1, y1, x2, y2 = hour_hand[0]
    cv2.line(img, (x1, y1), (x2, y2), (127, 0, 255), 3)
    cv2.putText(img, 'Hour', (int(x1), int(y1)), cv2.FONT_HERSHEY_TRIPLEX  , 1, (127,0,255), 2)

    # Draw line and add label for minute hand
    x1, y1, x2, y2 = minute_hand[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 102, 0), 3)
    cv2.putText(img, 'Minute', (int(x1), int(y1)), cv2.FONT_HERSHEY_TRIPLEX  , 1, (0,102,0), 2)

    # Draw line and add label for second hand
    x1, y1, x2, y2 = second_hand[0]
    cv2.line(img,(x1,y1),(x2,y2), (0,102,102), 3)
    cv2.putText(img, 'Second', (int(x1), int(y1)), cv2.FONT_HERSHEY_TRIPLEX  , 1, (0,102,102), 2)

# This function calculates the angle of the clock hands
def get_angle(hand, center_x, center_y):
    # Define the direction of the clock hands as a vector
    x1, y1, x2, y2 = hand[0]
    u = [x2 - x1, y2 - y1]
    # Define the horizontal direction as a vector
    v = [center_x - center_x, center_y - (center_y-100)]
    # Calculate the dot product of two vectors
    dot_uv = u[0] * v[0] + u[1] * v[1]
    # Calculate the length of the two vectors
    length_u = math.sqrt(u[0]**2 + u[1]**2)
    length_v = math.sqrt(v[0]**2 + v[1]**2)

    # Apply cosine formula to calculate the angle between two vectors
    cosine = dot_uv / (length_u * length_v)
    # Ensure that the cosine value is within the range [-1, 1]
    cosine = max(min(cosine, 1.0), -1.0)
    # Calculate the angle by applying the arc cosine function
    thet = math.acos(cosine)
    # Convert the angle from radians to degrees
    degrees = math.degrees(thet)
    # Calculate the cross product of two vectors
    cross_uv = u[0] * v[1] - u[1] * v[0]
    # If the directional product is greater than 0, that means vector u is to the left of vector v
    if cross_uv > 0:
        return 360 - degrees
    # If the directional product is less than 0, that means vector u is to the right of vector v
    else:
        return degrees
    
# This function calculates the time based on the angle of the clock hands
def get_time(hour_angle, minute_angle, second_angle):
    # Hour is calculated by dividing the angle of the hour hand by 30 (each hour corresponds to 30 degrees)
    hour = hour_angle / 30
    # Minute is calculated by dividing the angle of the minute hand by 6 (each minute corresponds to 6 degrees)
    minute = minute_angle / 6
    # Second is calculated by dividing the angle of the second hand by 6 (each second corresponds to 6 degrees)
    second = second_angle / 6
    # If the angle of the hour hand is close to an integer multiplied with 30 (i.e. close to a specific hour) and the angle of the minute hand is approximately between 0 and 6 (i.e. 1 round of 60 minutes has passed).
    if (round(hour)*30 - hour_angle <= 6) and ((355 < minute_angle and minute_angle < 360) or (minute_angle < 90)):
        hour = round(hour)
        # If the hour is 12, set it to 0
        if hour == 12:
            hour = 0
    
    # If the angle of the hour hand is close to a specific hour and the angle of the minute hand is close to 360 (ie close to 12 o'clock)
    if (hour_angle - hour*30 <= 6) and (355 < minute_angle and minute_angle < 360):
        # Set minute to 0
        minute = 0

    # If the angle of the minute hand is close to an integer multiplied by 6 (ie close to a specific minute) and the angle of the second hand is approximately between 0 and 6 (ie 1 round of 60 seconds has passed)
    if (round(minute)*6 - minute_angle <= 6) and (second_angle < 6):
        #Set second to 0
        minute = round(minute)
        if minute == 60:
            minute = 0

    # If the angle of the minute hand is close to a specific minute and the angle of the second hand is close to 360 (ie close to 60 seconds)
    if (minute_angle - minute*30 <= 6) and (354 < second_angle and second_angle < 360):
        # Set second to 0
        second = 0
    # Convert the hour, minute, and second to integers
    hour = int(hour)
    minute = int(minute)
    second = int(second)

    # Format the time in the form of HH:MM:SS
    time = f"{hour:02d}:{minute:02d}:{second:02d}"\

    return time

# This function display time on the image
def display_time(img, time):
    time = "Time:" + time
    # Font type, size, and thickness
    font = cv2.FONT_HERSHEY_TRIPLEX  
    scale = 2
    thickness = 3
    # Where to put the text on the image
    position = (50, 100)
    # Color of the text
    color = (255, 0, 255)
    # Draw the text on the image
    cv2.putText(img, time, position, font, scale, color, thickness)

def handle_step(img):
    # First step: pre-process the image to improve the quality of the image
    img = pre_process(img)
    # Convert image from BGR color space to HSV
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # Invert color values in HSV space
    img_hsv = cv2.bitwise_not(img_hsv)  
    # Balance the brightness of the image using the CLAHE method
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Use the CLAHE method to adjust the brightness of the V channel in the HSV image
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    # Use thresholding to segment the image
    _, thresh = cv2.threshold(img_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Use Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Second step: detect the clock by circle size or rectangle size
    center_x, center_y, radius = clock_detector(img, blurred)
    
    # Third step: detect the lines in the clock image
    lines = line_detector(img, blurred)

    # Fourth step: group lines that are close together and nearly parallel to each other
    groups = group_lines(lines, center_x, center_y, radius)

    # Next step: determine the clock hands
    hands = hands_detector(groups, center_x, center_y)

    # Next step: determine the hour hand, minute hand, and second hand
    hour_hand, minute_hand, second_hand = determine_hands(hands)

    # Next step: draw a frame around and label the clock hands back on the image
    draw_hands(img, hour_hand, minute_hand, second_hand)

    # Then calculate the angle of the clock hands
    hour = get_angle(hour_hand, center_x, center_y)
    minute = get_angle(minute_hand, center_x, center_y)
    second = get_angle(second_hand, center_x, center_y)

    # Next step: calculate the time based on the angle of the clock hands
    time = get_time(hour, minute, second)

    # Finally, display the time on the image
    display_time(img, time)

    return img
    
def main(input_dir, output_dir):
    # Step through the clock images from 1 to 10
    for i in range(1,11): 
        # File containing the clock image
        filename = f'clock{i}.jpg' 
        # Get the path to the image file
        img_path = os.path.join(input_dir, filename) 
        # If the image file does not exist
        if not os.path.exists(img_path): 
            filename = f'clock{i}.png' #Get the path to the image file with .png file extension
            img_path = os.path.join(input_dir, filename)
            if not os.path.exists(img_path): # If the image file does not exist, skip this file
                continue  
        
        #Read the image file
        img = cv2.imread(img_path)
        
        #Handle the image with the handle_step function
        img = handle_step(img)
        
        #Save the image to the output directory
        result_path = os.path.join(output_dir, f"{filename}")
        cv2.imwrite(result_path, img)
        
        #Display the image
        cv2.imshow(filename, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main('images','output_images')
