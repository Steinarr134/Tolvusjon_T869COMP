import random

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
        

