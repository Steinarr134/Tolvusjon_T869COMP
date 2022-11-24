import random
import numpy as np

def ransac(points, e, threshold=0.15, max_iter=30):
    """
    Uses ransac to find the most prominent line,

    returns the line coordinates and the indexes of the points that were inliers
    """
    # initialize variables

    best_score = 0
    best_line = None
    best_line_inliers = None

    n = len(points)*threshold

    for i in range(max_iter):

        # choose two points at random
        a = random.choice(points)
        b = random.choice(points)

        # todo: fix edge case where a==b

        # Calculate the line, y=mx + k between them
        m = (a[1] - b[1])/(a[0] - b[0])
        k = a[1] - m*a[0]

        # count number of points closer than e to the line
        inliers = []
        for i, (x, y) in enumerate(points):
            err = y - (m*x + k)
            if abs(err) < e:
                inliers.append(i)
        
        if len(inliers) > n:
            return (m, k), inliers
        else:
            if best_score < len(inliers):
                best_score = len(inliers)
                best_line = (m, k)
                best_line_inliers = inliers
                
    return best_line,best_line_inliers


if __name__ == '__main__':
    import cv2
    import time
    def tu(a):
        return tuple((int(a[1]), int(a[0])))
    frame = cv2.imread("frame.jpg")

    frame = cv2.resize(frame[:4000, :], (480, 640))
    t0 = time.time()
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    medianed = cv2.medianBlur(blurred, 9)
    gray = cv2.cvtColor(medianed, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray", gray)
    canny = cv2.Canny(gray, 50, 200)

    cv2.imshow("Canny", canny)

    points = np.argwhere(canny)

    colored_canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    for i in range(4):

        line, inliers = ransac(points, 4)

        cv2.line(colored_canny, tu((0, int(line[1]))), tu((1000, int(1000*line[0] + line[1]))), (0, 255, 0), 3)
        print(len(points), len(inliers))

        points = np.delete(points, inliers, 0)
        print(len(points))
    cv2.imshow("ransac", colored_canny)

    cv2.waitKey()

    print(line, len(inliers))
