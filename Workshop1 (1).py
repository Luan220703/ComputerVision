import cv2 as cv
import numpy as np 

# Create a Rectangle Class
class Rectangle:
    def __init__(self, points):
        self.p1 = points[0]
        self.p3 = points[1]
        self.p2 = [self.p3[0], self.p1[1]]
        self.p4 = [self.p1[0], self.p3[1]]
        self.mid = [(self.p1[0] + self.p3[0]) / 2, (self.p1[1] + self.p3[1]) / 2]
        self.arr = [self.p1, self.p2, self.p3, self.p4]
        
    def update_coordinates(self, arr):
        self.arr = arr
        self.p1 = arr[0]
        self.p2 = arr[1]
        self.p3 = arr[2]
        self.p4 = arr[3]
        self.mid = [(self.p1[0] + self.p3[0]) / 2, (self.p1[1] + self.p3[1]) / 2]


# Create a white background
def create_white_bg():
    global blank
    blank = np.ones((750, 750, 3), dtype="uint8") * 255

# Draw a rectangle based on 2 coordinate p1(x1, y1) and p2(x2, y2)
def draw_rectangle(event, x, y, flags, param):
    global points, rec, blank, drawing
    blank = np.ones((750, 750, 3), dtype="uint8") * 255
    
    if event == cv.EVENT_LBUTTONDOWN:
        points = [[x, y]]
        drawing = True
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if points[0][0] != x or points[0][1] != y:
                blank = np.ones((750, 750, 3), dtype="uint8") * 255
                cv.rectangle(blank, points[0], [x, y], (255, 0, 0), 2)
                cv.imshow("Window", blank)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        points.append([x, y])
        rec = Rectangle(points)
        cv.rectangle(blank, points[0], points[1], (255, 0, 0), -1)
        cv.imshow("Window", blank)

# Translation transformation
def translate(x, y, arr):
    global blank
    blank = np.ones((750, 750, 3), dtype="uint8") * 255
    newArr = []
    for point in arr:
        newArr.append([point[0] + x, point[1] + y])
    
    rec.update_coordinates(newArr)
    cv.fillPoly(blank, pts=np.int32([newArr]), color=(255, 0, 0))
    cv.imshow("Window", blank)

# Rotation transformation
def rotatePoint(point, angle):
    x, y = point
    oX, oY = rec.mid
    angleToRadian = angle * np.pi / 180
    rX = oX + (x - oX) * np.cos(angleToRadian) - (y - oY) * np.sin(angleToRadian)
    rY = oY + (x - oX) * np.sin(angleToRadian) + (y - oY) * np.cos(angleToRadian)
    return [rX, rY]

def rotateRec(arr, angle):
    global blank
    newArr = []
    for point in arr:
        newArr.append(rotatePoint(point, angle))

    rec.update_coordinates(newArr)
    blank = np.ones((750, 750, 3), dtype="uint8") * 255
    cv.fillPoly(blank, pts=np.int32([newArr]), color=(255, 0, 0))
    cv.imshow("Window", blank)
    
# Scaling transformation
def scaleRec(arr, sx, sy):
    global blank
    newArr = []
    for point in arr:
        newArr.append([sx * point[0], sy * point[1]])

    rec.update_coordinates(newArr)
    blank = np.ones((750, 750, 3), dtype="uint8") * 255
    cv.fillPoly(blank, pts=np.int32([newArr]), color=(255, 0, 0))
    cv.imshow("Window", blank)


# Menu
def returnOption():
    print("-------------------------------------------------------")
    print("1. Create a white background.")
    print("2. Draw rectangle.")
    print("3. Translation.")
    print("4. Rotation.")
    print("5. Scaling.")
    print("6. Exit.")
    option = input("Enter an options: ")
    if option in ["1", "2", "3", "4", "5", "6"]:
        return option
    print("Invalid option! Try again.")
    return False
       
       
       
if __name__ == "__main__":
    points = []
    drawing = False

    while True:
        choice = int(returnOption())
        if not choice:
            continue
        elif choice == 1:
            create_white_bg()
            cv.imshow("Window", blank)
        elif choice == 2:
            cv.setMouseCallback("Window", draw_rectangle)
        elif choice == 3:
            tx = int(input("Enter the translation of X: "))
            ty = int(input("Enter the translation of Y: "))
            translate(tx, ty, rec.arr)
        elif choice == 4:
            angle = float(input("Enter the angle of rotation: "))
            rotateRec(rec.arr, angle)
        elif choice == 5:
            sx = float(input("Enter the scale of x coordinate: "))
            sy = float(input("Enter the scale of y coordinate: "))
            scaleRec(rec.arr, sx, sy)
        elif choice == 6:
            break
        cv.waitKey(0)
    
        