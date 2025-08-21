from PIL import Image
import numpy as np
from heapq import heappush as push, heappop as pop
import math

img = Image.open("circle_border.png")
arr = np.array(img) #define the image to work on.
#——————————————————————————————Contour Tracing—————————————————————————————
def is_valid(G, r, c, threshold=50):
    """Return True if pixel is black and has a neighbor above threshold."""
    if not np.array_equal(G[r, c], [0,0,0,255]):
        return False
    for dr, dc in [(-1,-1), (1,-1), (1,1), (-1,1), (-1,0), (0,-1), (0,1), (1,0)]:
        if np.any(G[r+dr, c+dc][:3] > threshold):
            return True
    return False

def find_start(G, threshold=50):
    """Find the first valid pixel to begin tracing."""
    rows, cols = G.shape[:2]
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if is_valid(G, r, c, threshold):
                return (r, c)
    return None

def ordered_contour(G, threshold=50):
    """
    Return an ordered contour pixel list from image array G.
    """
    directions = [
        (0,1), (1,0), (1,1), (0,-1),
        (1,-1), (-1,0), (-1,-1), (-1,1)
    ]
    
    start = find_start(G, threshold)
    if not start:
        return []

    path, visited = [], set()
    r, c = start
    while True:
        visited.add((r, c))
        path.append((r, c))
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if is_valid(G, nr, nc, threshold) and (nr, nc) not in visited:
                r, c = nr, nc
                break
        else:
            break  # no neighbor found
    return path


# ———————————————————————————— Contour Tracing ———————————————————
#===================================================================
# ————————————————————————————— Dijikstra ——————————————————————————
#===================================================================
def minigraph(pixel_list, s, e):
    """Create a minigraph around pixel_list[s] and pixel_list[e]
        Paramters:-
            pixel_list: a list of pixels in the form of (r, c)
            s: start coords in the form of (r, c)
            e: end coords in the form of (r, c)
        Output:-
            minigrid: an array containing all pixels in our minigraph
    """
    subset = pixel_list[s:e + 1]

    rows = [r for r, _ in subset]
    cols = [c for _, c in subset]

    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    mingrid  = []
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            mingrid.append((r, c))

    return mingrid 

def dijkstra(G, pixel_list, i, n):
    """
    Perform modified Dijkstra from pixel_list[i] to pixel_list[i+n].
    Paramters:-
        G: 2D numpy array
        pixel_list: a list of pixels in the form of (r, c)
        i: index/start point
        n: integer value for sets of length n.
    Output:
        Shortest straight path from pixel_list[i] -> pixel_list[i+n]
    """
    s = i
    e = min(i + n, len(pixel_list) - 1)

    start = pixel_list[s]
    end = pixel_list[e]
    ideal_r, ideal_c = start

    min_graph = minigraph(pixel_list, s, e)
    pq = []
    dist = {}
    parent = {}
    visited = set()
    
    # Initialize distances and parents
    for r, c in min_graph:
        dist[(r, c)] = float('inf')
        parent[(r, c)] = None
        G[r, c] = [255, 255, 255, 255]  # remove old path
    
    dist[start] = 0
    push(pq, (0, start))

    while pq:
        current_dist, (r, c) = pop(pq)
        if (r, c) in visited:
            continue
        visited.add((r, c))

        if (r, c) == end:
            break

        # 4-neighbors only don't want diagonal path only straight paths
        for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]:
            if (nr, nc) in min_graph and (nr, nc) not in visited:

                #current weight system, straight paths from s->e are prioritized
                weight = abs(nr - ideal_r) + abs(nc - ideal_c)
                new_dist = current_dist + weight


                if new_dist < dist[(nr, nc)]:
                    dist[(nr, nc)] = new_dist
                    parent[(nr, nc)] = (r, c)
                    push(pq, (new_dist, (nr, nc)))

    # Reconstruct path and draw
    path = []
    pixel = end
    while pixel is not None:
        path.append(pixel)
        pixel = parent[pixel]
    path.reverse()
    traverse(path, G)
    return path

def traverse(path, G):
    """Draw path in black on image G."""
    for r, c in path:
        G[r, c] = [0, 0, 0, 255]

def apply_dijkstra_all(G, pixel_list, n):
    """Apply Dijkstra on every n-step segment of pixel_list."""
    full_path = []
    for i in range(0, len(pixel_list)-1, n):
        path = dijkstra(G, pixel_list, i, n)
        full_path.extend(path)
    return full_path

#—————————————————— Dijikstra —————————————————————————————
#===========================================================
#—————————————————— Chaikin —————————————————————————————————

def find_vertices(contour_list, min_dist=2):
    """ Finds the vertices from contour_list
        Paramters:-
            contour_list: list of pixels in the form of (r, c)
            min_dist: optional parameter so that each vertex is min_dist away from the next
        Outputs:-
            vertices: an array of all vertices in the form of (r, c)
    """
    vertices = []
    prev_dir = None
    last_vertex = None

    for i in range(len(contour_list) - 1):
        r1, c1 = contour_list[i]
        r2, c2 = contour_list[i + 1]
        curr_dir = (r2 - r1, c2 - c1)

        #if change in direction then it's a vertex
        if prev_dir is None or curr_dir != prev_dir:
            if last_vertex is None:
                vertices.append((r1, c1))
                last_vertex = (r1, c1)
            else:
                #there must be min_dist between vertex A and vertex B.
                dist = ((r1 - last_vertex[0]) ** 2 + (c1 - last_vertex[1]) ** 2) ** 0.5
                if dist >= min_dist:
                    vertices.append((r1, c1))
                    last_vertex = (r1, c1)
        prev_dir = curr_dir

    return vertices

def lerp(p1, p2, r):
    """Calculates linear interpolation from p1 onto p2 with ratio r
    """
    return p1 * (1 - r) + p2 * r


def chaikin_cut(v1, v2, ratio):
    """Performs a chaikin cut on vertex v1 and vertix v2 with input ratio.
        Paramters:-
          v1, v2: vertex coordinates in the form of (r, c)
          ratio: float value 0 ≤ ratio ≤ 1
        Outputs:
            cut1, cut2: the cut coordinates for v1 and v2 respectively.
    """

    if ratio > 0.5: ratio = 1 - ratio

    v1_r = lerp(v1[0], v2[0], ratio)
    v1_c = lerp(v1[1], v2[1], ratio)
    cut1 = (v1_r, v1_c)

    v2_r = lerp(v2[0], v1[0], ratio)
    v2_c = lerp(v2[1], v1[1], ratio)
    cut2 = (v2_r, v2_c)

    return cut1, cut2


def draw_polyline(points, G, color=[0,0,0,255], closed=False):
    """Connects points together on G.
        
        Parameters:
            points: tuple array of (r, c)
            G: 2D numpy array
            color: RGBA color
            closed: if True, connect last point back to first point

        Output:
            connected points on graph G.
    """
    if len(points) < 2:
        return
        
    # Draw lines between consecutive points
    for i in range(len(points) - 1):
        connect(points[i], points[i+1], G, color)
    
    # If closed connect the last point back to the first
    if closed and len(points) > 2:
        connect(points[-1], points[0], G, color)

def connect(p1, p2, G, color=[0,0,0,255]):
    """
    Draw a straight line between p1 and p2 on arr using Bresenham's line algorithm.

    Parameters:
        p1, p2 : tuple of integers (r,c)
        G: 2D numpy array

    Output:
        Straight line from p1->p2 on graph G
    """
    # Round and convert coordinates to integers
    r0 = int(round(p1[0]))
    c0 = int(round(p1[1]))
    r1 = int(round(p2[0]))
    c1 = int(round(p2[1]))

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1

    err = dr - dc

    while True:
        G[r0][c0] = color
        if r0 == r1 and c0 == c1:
            break
        e2 = err * 2
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


def chaikin(vertices, ratio=0.25, iterations=1, closed=False):
    """
    Smooth a structuer using Chaikin's corner cutting algorithm.
    
    Parameters:
        vertices: list of (x, y)
        ratio: how far in to cut from each edge (0 < ratio < 0.5 typical)
        iterations: number of smoothing passes
        closed: if True, treat vertices as a closed polygon
    Output:
        result: an ordered array of (r, c) that stores the post-cut structure.
    """
    if len(vertices) < 2:
        return vertices[:]  # No smoothing possible
    
    result = vertices[:]
    for _ in range(iterations):
        new_points = []
        n = len(result)

        if closed:
            # loop over every pair, including last->first
            for i in range(n):
                p0 = result[i]
                p1 = result[(i + 1) % n]
                Q, R = chaikin_cut(p0, p1, ratio)
                new_points.extend([Q, R])
            result = new_points
        else:
            # Keep endpoints, smooth in between
            new_points.append(result[0])
            for i in range(n - 1):
                p0 = result[i]
                p1 = result[i + 1]
                Q, R = chaikin_cut(p0, p1, ratio)
                new_points.extend([Q, R])
            new_points.append(result[-1])
            result = new_points
    
    return result

def slice_between_vertices(vertices, a, b, closed=True):
    """
    Return the sublist of vertices from a to b in order.
    
    Parameters:
        vertices: List of vertices in form of (r, c)
        a: Starting vertex.
        b: Ending vertex.
        closed: If True, allows wrapping around the end of the list.
    
    Outputs:
        A list of vertices from a to b inclusive.
    """
    # just standard safety
    if a not in vertices or b not in vertices:
        return None

    ia = vertices.index(a)
    ib = vertices.index(b)

    if ia <= ib:
        seg = vertices[ia:ib+1]
    else:
        if not closed:
            return None
        seg = vertices[ia:] + vertices[:ib+1]
    return seg
#———————————————————— Chaikin ———————————————————————–
#=====================================================
#————————————————————— RDP ———————————————————————————

def perpendicular_distance(point, start, end):
    """ 
    Calculates perpendicular distance for coordinate point from start to end.
    """
    x0, y0 = start
    x1, y1 = end
    x, y = point

    # If start and end are the same point, just return distance to start
    if x0 == x1 and y0 == y1:
        return math.hypot(x - x0, y - y0)

    return abs((x1 - x0)*(y0 - y) - (x0 - x)*(y1 - y0)) / math.hypot(x1 - x0, y1 - y0)

def RDP(vertices, epsilon):
    """
    Ramer Douglas Peucker algorithm. Obtain striking features from vertices.
    
    Paramters:
        vertices: list of (r,c) points
        epsilon: minimum perpendicular distance to keep a point
    
    Output:
        array of striking features in form of (r, c)
    """
    if len(vertices) < 3:
        return vertices

    start, end = vertices[0], vertices[-1]
    max_dist = 0
    index = 0

    # Find point with max distance from line segment
    for i in range(1, len(vertices) - 1):
        dist = perpendicular_distance(vertices[i], start, end)
        if dist > max_dist:
            max_dist = dist
            index = i

    # If max distance is greater than epsilon, keep the point and recurse
    if max_dist > epsilon:
        left = RDP(vertices[:index+1], epsilon)
        right = RDP(vertices[index:], epsilon)
        return left[:-1] + right
    else:
        # Otherwise,
        return [start, end]

#————————————————————————— RDP ————————————————————————————
#==========================================================
#————————————————————————— Testing ————————————————————————


pixel_list = ordered_contour(arr.copy()) #contour
dij = apply_dijkstra_all(arr.copy(), pixel_list, 7) #exaggerates features in contour

#===============================
# STEP 1: Dijkstra only
#===============================
step1_img = np.full_like(arr, [255, 255, 255, 255])  # clean white
for (r, c) in dij:
    step1_img[r][c] = [0, 0, 0, 255]  # black path
Image.fromarray(step1_img).save("Step_1.png")

# get vertices from Dijkstra path
vertices = find_vertices(dij, min_dist=10)

# RDP to serve as pillars prep for step 2
pillars = RDP(vertices, epsilon=5)

#===============================
# STEP 2: Dijkstra + RDP pillars
#===============================
step2_img = step1_img.copy()
for (r, c) in pillars:
    step2_img[r][c] = [255, 0, 0, 255]  # red pillars
Image.fromarray(step2_img).save("Step_2.png")

#===============================
# STEP 3: Chaikin smoothed using RDP pillars
#===============================
step3_img = np.full_like(arr, [255, 255, 255, 255])  # clean white

smoothed_vertices = []
closed_contour = True
pairs = [(pillars[i], pillars[i+1]) for i in range(len(pillars)-1)]
if closed_contour:
    pairs.append((pillars[-1], pillars[0]))

# apply chaikin on each segment to maintain integrity of each pillar
for i, (seg_start, seg_end) in enumerate(pairs):
    segment = slice_between_vertices(vertices, seg_start, seg_end, closed=closed_contour)
    if not segment or len(segment) < 2:
        continue

    smoothed_segment = chaikin(segment, ratio=0.25, iterations=5)

    if i > 0:
        smoothed_vertices.extend(smoothed_segment[1:])
    else:
        smoothed_vertices.extend(smoothed_segment)

draw_polyline(smoothed_vertices, step3_img)
Image.fromarray(step3_img).save("Step_3.png")

#===============================
# STEP 4: All steps combined
#===============================
all_img = np.full_like(arr, [255, 255, 255, 255])  # start with white background


COLOR_ORIGINAL = (150, 150, 150, 180)   # light grey
COLOR_DIJKSTRA = (30, 60, 120, 255)     # dark navy blue
COLOR_PILLARS  = (255, 120, 0, 255)     # bright orange
COLOR_CHAIKIN  = (0, 200, 120, 255)     # cyan-green

# 1. Original contour
for (r, c) in pixel_list:
    all_img[r][c] = COLOR_ORIGINAL

# 2. Dijkstra path
for (r, c) in dij:
    all_img[r][c] = COLOR_DIJKSTRA

#3. Chaikin path
draw_polyline(smoothed_vertices, all_img, color=COLOR_CHAIKIN)

# 4. RDP pillars
for (r, c) in pillars:
    all_img[r][c] = COLOR_PILLARS


Image.fromarray(all_img).save("All_Steps.png")

#===============================
# STEP 5: Before & After
#===============================
before_after = np.full_like(arr, [255, 255, 255, 255])  # white background

COLOR_CONTOUR = (150, 150, 150, 255)  # grey
COLOR_CHAIKIN = (0, 0, 0, 255)        # black

# Draw original contour in grey
for (r, c) in pixel_list:
    before_after[r][c] = COLOR_CONTOUR

# Draw Chaikin path in black
draw_polyline(smoothed_vertices, before_after, color=COLOR_CHAIKIN)

# Save
Image.fromarray(before_after).save("Before_After.png")


#===============================
# STEP 6: Before & After (Chaikin on vertices of pixel_list)
#===============================
before_after = np.full_like(arr, [255, 255, 255, 255])  # white background

COLOR_CONTOUR = (150, 150, 150, 255)  # grey
COLOR_CHAIKIN = (0, 0, 0, 255)        # black



# 2. Extract vertices from pixel_list
vertices_pixel_list = find_vertices(pixel_list, min_dist=5)  # tweak min_dist as needed

# 3. Apply Chaikin on vertices
smoothed_vertices_direct = chaikin(vertices_pixel_list, ratio=0.25, iterations=3, closed=True)

# 4. Draw Chaikin result in black
draw_polyline(smoothed_vertices_direct, before_after, color=COLOR_CHAIKIN, closed=True)

# Save
Image.fromarray(before_after).save("Before_After2.png")