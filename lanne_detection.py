#!/usr/bin/env python3
# Import only if not previously imported
import cv2
import numpy as np
# In VideoCapture object either Pass address of your Video file
# Or If the input is the camera, pass 0 instead of the video file
cap = cv2.VideoCapture('mumbai.mp4')


def get_coordinates(image, params):

    slope, intercept = params
    # print(slope, intercept)
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))  # Setting y2 at 3/5th from y1
    x1 = int((y1 - intercept) / slope)  # Deriving from y = mx + c
    x2 = int((y2 - intercept) / slope)
    # print(np.array([x1, y1, x2, y2]))
    return np.array([x1, y1, x2, y2])

# Returns averaged lines on left and right sides of the image


def avg_lines(image, lines):

    left = []
    right = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            # Fit polynomial, find intercept and slope
            params = np.polyfit((x1, x2), (y1, y2), 1)
            slope = params[0]
            y_intercept = params[1]

            if slope < 0:
                # Negative slope = left lane
                left.append((slope, y_intercept))
            else:
                # Positive slope = right lane
                right.append((slope, y_intercept))

    # Avg over all values for a single slope and y-intercept value for each line

        left_avg = np.average(left, axis=0)
        right_avg = np.average(right, axis=0)
        # print(left_avg, right_avg)
        # Find x1, y1, x2, y2 coordinates for left & right lines
        if (str(left_avg) != 'nan') and (str(right_avg) != 'nan'):
            right_line = get_coordinates(image, right_avg)
            left_line = get_coordinates(image, left_avg)
            return np.array([left_line, right_line])


def display_lines(image, lines):

    if lines is not None:
        print(lines)
        for x1, y1, x2, y2 in lines:
            cv2.line(image, (x1, y1),
                     (x2, y2), (255, 0, 0), 2)
    return image


if cap.isOpened() == False:
    print("Error in opening video stream or file")
while(cap.isOpened()):

    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        # Import only if not previously imported

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (5, 5), 0)

    # # Canny edge detector with minVal of 50 and maxVal of 150
        img = cv2.Canny(blur, 50, 150)
        stencil = np.zeros(img.shape, dtype=img.dtype)

        # specify coordinates of the polygon
        polygon = np.array(
            [[50, 480], [416, 289], [530, 271],  [854, 400], [854, 480]])

        # fill polygon with ones
        cv2.fillConvexPoly(stencil, polygon, 255)
        image = cv2.bitwise_and(img, img, mask=stencil)
        ret, thresh = cv2.threshold(image, 130, 145, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180,
                                30, minLineLength=30,  maxLineGap=200, lines=np.array([]))

        # create a copy of the original frame
        dmy = frame

        # # draw Hough lines
        if lines is not None:

            # print(lines)
            for line in lines:

                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)
                params = np.polyfit((x1, x2), (y1, y2), 1)
                slope = params[0]
                y_intercept = params[1]
                # print(line[0])
                if slope < 0:
                    # Negative slope = left lane
                    left = line[0]

                else:
                    # Positive slope = right lane

                    right = line[0]

        final = [left] + [right]
        print(np.array(final))
        # if lines is not None:
        #     averaged_lines = avg_lines(frame, lines)
        # frame2 = display_lines(frame, averaged_lines)
        # cv2.imshow('Frame', image)
        cv2.imshow('mask', dmy)
        # Press esc to exit
        if cv2.waitKey(20) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
