import cv2
import numpy as np
import math

def resize(frame, out_w, out_h):
    """Returns: scaled image of given width and height"""
    h, w = frame.shape[:2]
    #Create a NumPy array with evenly spaced values in a given interval and index frame to that array
    return frame[
        (np.arange(out_h) * h // out_h)[:, None],
        (np.arange(out_w) * w // out_w)[None, :]
    ]

def extract_boundary(binary_frame):
    """Returns: binary image with boundaries as 1 / white / 255"""
    
    #STEP1- Convert binary(0/255) or binary(0/1) binary(True/False)
    b = binary_frame.astype(bool)

    #STEP2- Perform 3x3 erosion logically (apply AND: if all 9 are 1 i.e TRUE set TRUE) A pixel stays ON only if it and all 8 neighbors are ON.
    eroded = (
        b[1:-1, 1:-1] &
        b[:-2, 1:-1] &
        b[2:, 1:-1] &
        b[1:-1, :-2] &
        b[1:-1, 2:] &
        b[:-2, :-2] &
        b[:-2, 2:] &
        b[2:, :-2] &
        b[2:, 2:]
    )

    #STEP3- Pad back with original binary
    eroded_full = np.zeros_like(b)
    eroded_full[1:-1, 1:-1] = eroded

    #STEP4- boundary = binary AND NOT(eroded)
    boundary = b & (~eroded_full)

    return boundary.astype(np.uint8) * 255

def get_connected_components(binary,scaling):
    """
    binary: 2D numpy array (0 and 255 or 0 and 1)
    returns: list of components (each component is list of (r, c))
    """
    h, w = binary.shape

    #STEP1- Convert to boolean for faster neighbour checking
    img = (binary != 0)
    #INITIALIZE
    visited = np.zeros((h, w), dtype=np.uint8)
    components = []

    # 8-connectivity offsets
    neighbors = [(-1,-1), (-1,0), (-1,1),
                 (0,-1),          (0,1),
                 (1,-1),  (1,0),  (1,1)]
    #STEP2- LOOP THROUGH PIXELS AND STACK CONNECTED COMPONENTS
    for c in range(h):
        for r in range(w):
            #FOR EACH WHITE PIXEL NOT VISITED
            if img[c, r] and not visited[c, r]:

                stack = [(r, c)]
                visited[c, r] = 1
                component = [(r, c)]

                #BUILD A STACK OF UNVISITED BUT EXPLORED COMPONENTS AND LOOP UNTIL STACK IS EMPTY
                while stack:
                    cr, cc = stack.pop()

                    for dr, dc in neighbors:
                        nr = cr + dr
                        nc = cc + dc
                        #IF NEIGHBOUR IS WHITE AND NOT VISITED
                        if 0 <= nc < h and 0 <= nr < w:
                            if img[nc, nr] and not visited[nc, nr]:
                                visited[nc, nr] = 1
                                stack.append((nr, nc))
                                component.append((nr, nc))

                components.append(component)
    
    #SCALE FOR ORIGINAL IMAGE
    scaled = []
    for cc in components:
        arr = np.asarray(cc, dtype=np.float32)
        arr[:, 0] *= scaling
        arr[:, 1] *= scaling
        arr = arr.astype(np.int32)
        scaled.append(arr)

    return scaled

def draw_contours(img, color, contours): 
    """ Overlay contours on an image """ 
    vis = img.copy()
    for contour in contours:
        for x, y in contour:
            vis[y, x] = color
            for (dx,dy) in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                if y+dy>=0 and y+dy<vis.shape[0] and x+dx>=0 and x+dx<vis.shape[1]:
                    vis[y+dy, x+dx] = color
    return vis

def get_quad_corners(component):
    """
    Finds the 4 corners of a quadrilateral compoenet using the 
    Longest Diagonal + Furthest Point method.
    Returns ordered corners [TL, BL, BR, TR]
    """
    pts = np.array(component)
    if len(pts) < 4:
        return []

    # STEP1. Find two points with maximum distance (The Main Diagonal)
    # Optimization: Instead of O(N^2) distance check, we iterate relative to a few pivots
    # Pivot A: Point furthest from centroid
    # Pivot B: Point furthest from A (Likely one corner)
    # Pivot C: Point furthest from B (Likely the opposite corner)
    
    # We use squared Euclidean distance
    def get_farthest_index(pt, points):
        d2 = np.sum((points - pt) ** 2, axis=1)
        return np.argmax(d2)

    # Start with centroid, find corner 1
    centroid = pts.mean(axis=0)
    idx1 = get_farthest_index(centroid, pts)
    p1 = pts[idx1]

    # Find corner 2 (opposite to p1)
    idx2 = get_farthest_index(p1, pts)
    p2 = pts[idx2]

    #NOW WE HAVE 2 of the 4 corners from the longest diagonal
    # STEP2. Form the diagonal vector (Line P1 -> P2)
    # Line equation: cross_product((p - p1), (p2 - p1)) to get perpendicular distance
    # This gives the signed distance (scaled) from the line
    vec = p2 - p1
    
    # Vector from p1 to all other points
    vec_pts = pts - p1
    
    # 2D Cross Product: A_x * B_y - A_y * B_x
    # (N, 2) cross (2,) -> (N,)
    cross_prods = vec_pts[:, 0] * vec[1] - vec_pts[:, 1] * vec[0]

    # STEP3. Find the extrema on both sides of the line
    # One corner will have max positive cross product
    # One corner will have min negative cross product
    idx3 = np.argmax(cross_prods)
    idx4 = np.argmin(cross_prods)

    # We now have 4 indices: idx1, idx2, idx3, idx4
    # Note: If the shape is a degenerate line, idx3 or idx4 might equal idx1/idx2.
    # We should filter unique.
    corner_indices = np.unique([idx1, idx2, idx3, idx4])
    if len(corner_indices) != 4:
        # Fallback: The component might be a triangle or extremely noisy line
        return []

    corners = pts[corner_indices].tolist()

    # STEP4. Order the corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    # Standard way: Sort by Y first, then X, or use centroid angles.
    #   TL = min(x+y), BR = max(x+y)
    #   TR = min(x-y), BL = max(x-y)
    # BUT: This simple sort fails on 45-degree rotation.

    # sum and difference
    sum_pts = [x + y for x, y in corners]
    diff_pts = [x - y for x, y in corners]

    tl = corners[sum_pts.index(min(sum_pts))]
    br = corners[sum_pts.index(max(sum_pts))]
    tr = corners[diff_pts.index(min(diff_pts))]
    bl = corners[diff_pts.index(max(diff_pts))]

    return [tl, bl, br, tr]

def quad_area(points):
    """
    points: list of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            must be ordered clockwise or counter-clockwise
    
    returns: area (float)
    """

    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    area = abs(
        x1*y2 + x2*y3 + x3*y4 + x4*y1
        - (y1*x2 + y2*x3 + y3*x4 + y4*x1)
    ) * 0.5

    return area

def compute_homography(corners,tagsize):
    """
    corners: list of 4 (x, y)
    returns: 3x3 homography matrix H
    """
    #STEP1 - set target points for corners
    target_points = [
                (0, 0),
                (tagsize - 1, 0),
                (tagsize - 1, tagsize - 1),
                (0, tagsize - 1)]
    
    #BUILD THE A and b matrices for Ah = b
    A = []
    b = []
    for (x, y), (u, v) in zip(corners, target_points):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        b.append(u)
        b.append(v)
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    # Solve Ah = b
    h = np.linalg.solve(A, b)

    #BUILD H MATRIX
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ])

    #RETURN THE INVERSE H MATRIX
    return H

def apply_inverse_homography(gray, H_inv, tagsize):
    """Applies H_inverse to give frame and returns warped image"""
    h_out, w_out = tagsize,tagsize
    h_src, w_src = gray.shape

    #STEP1- Create meshgrid for output coordinates (u horizontal, v vertical)
    u, v = np.meshgrid(np.arange(w_out), np.arange(h_out), indexing='ij')
    ones = np.ones_like(u)
    # Stack to homogeneous points: shape (h_out, w_out, 3)
    pts = np.stack([u, v, ones], axis=-1).reshape(-1, 3)  # (h_out*w_out, 3)

    #STEP2- Apply transformation: pts @ H_inv.T
    q = pts @ H_inv.T  # (N, 3)

    #STEP3- Normalize
    q[:, :2] /= q[:, 2:3]

    #STEP4- Manipulate and Bound
    # Round to nearest integers
    x = np.round(q[:, 0]).astype(int)
    y = np.round(q[:, 1]).astype(int)
    # Bounds mask
    mask = ((0 <= x) & (x < w_src) & (0 <= y) & (y < h_src)).flatten()
    # Gather values
    warped_flat = np.zeros(h_out * w_out, dtype=gray.dtype)
    valid_indices = mask.nonzero()[0]
    warped_flat[valid_indices] = gray[y[valid_indices], x[valid_indices]]
    # Reshape to output
    return warped_flat.reshape(h_out, w_out)

def verify_tag(binary, tagsize, scaling):
    unit = tagsize/8
    border = np.zeros_like(binary)
    border[:int(2*unit),:] = binary[:int(2*unit),:]
    border[int(6*unit):,:] = binary[int(6*unit):,:]
    border[:,:int(2*unit)] = binary[:,0:int(2*unit)]
    border[:,int(6*unit):] = binary[:,int(6*unit):]
    return np.mean(border)<25*scaling

def get_orientation(binary, tagsize):
    unit = tagsize/8
    tmp = binary.copy()
    tmp[int(2.5*unit):int(5.5*unit),:] = 0
    tmp[:,int(2.5*unit):int(5.5*unit)] = 0
    indicator_list = np.array([np.mean(tmp[int(5.5*unit):int(5.5*unit)+10,int(5.5*unit):int(5.5*unit)+10]),
                               np.mean(tmp[int(2.5*unit)-10:int(2.5*unit),int(5.5*unit):int(5.5*unit)+10]),
                               np.mean(tmp[int(2.5*unit)-10:int(2.5*unit),int(2.5*unit)-10:int(2.5*unit)]),
                               np.mean(tmp[int(5.5*unit):int(5.5*unit)+10,int(2.5*unit)-10:int(2.5*unit)])])
    # print(indicator_list)
    return np.argmax(indicator_list)

def decode_id(binary, rotation, tagsize):
    unit = tagsize/8
    tmp = binary.copy()
    tmp[:int(3*unit),:] = 255
    tmp[int(5*unit):,:] = 255
    tmp[:,0:int(3*unit)] = 255
    tmp[:,int(5*unit):] = 255
    bit_map = [tmp[int(3.1*unit):int(3.9*unit),int(3.1*unit):int(3.9*unit)],
            tmp[int(3.1*unit):int(3.9*unit),int(4.1*unit):int(4.9*unit)],
            tmp[int(4.1*unit):int(4.9*unit),int(4.1*unit):int(4.9*unit)],
            tmp[int(4.1*unit):int(4.9*unit),int(3.1*unit):int(3.9*unit)]]
    bits = [1,1,1,1]
    for i in range(4):
        bits[i] = int(np.mean(bit_map[i-rotation]))
    # print(bits)
    thresh = 0.9*(min(bits)+(max(bits)-min(bits))/4)
    for i in range(4):
        bits[i] = int(np.mean(bits[i])>=thresh)
    # print(bits,thresh)
    tag_id = (bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3]
    return tag_id

def decode_tag(binary_warped, tagsize):
    """
    RETURNS: number of rotations for target image placement, TAG_ID
    """
    rotation = get_orientation(binary_warped, tagsize)
    # print("rotation",rotation)
    tag_id = decode_id(binary_warped,rotation, tagsize)
    return rotation, tag_id

def task1(img, tagsize=120, scaling=2, global_threshold=165):
    """Task 1: AR-Tag Detection and Identification
        Return: corners, IDs and orientation of detected tag"""
    og_frame = img.copy()
    w ,h, _ = og_frame.shape
    detections = []
    #DOWNSCALE THE FRAME
    frame = resize(og_frame, int(h//scaling),int(w//scaling))
    #STEP1- Convert to grayscale
    gray_frame = (0.33*frame[:,:,0]+0.33*frame[:,:,1]+0.33*frame[:,:,2]).astype(np.uint8)
    og_gray = (0.33*og_frame[:,:,0]+0.33*og_frame[:,:,1]+0.33*og_frame[:,:,2]).astype(np.uint8)
    #STEP2- Convert to BINARY using Global Thresholding
    binary_frame = (gray_frame >= global_threshold).astype(np.uint8) * 255
    #STEP3- Extract boudary points by erosion
    bnd = extract_boundary(binary_frame)
    #STEP4- Get the boundaries by contouring
    connected_components = get_connected_components(bnd, scaling)
    # vis_frame = draw_contours(og_frame,[255,0,0],connected_components)
    #STEP5- FOR EACH COMPONENT IN CONNECTED_COMPONENTS
    for i,components in enumerate(connected_components):
        #DETECT the 4 corners by longest diagonal and perpendicular diagonal
        corners = get_quad_corners(components)
        if corners and quad_area(corners)>1000:
            # print("A",quad_area(corners))
            #STEP6- COMPUTE INVERSE HOMOGRAPHY 
            try:
                h = compute_homography(corners, tagsize)
                h_inv = np.linalg.inv(h)
            except:
                #INCASE OF SINGULARITY
                continue
            #STEP7 - APPLY THE INVERSE HOMOGRAPHY
            warped = apply_inverse_homography(og_gray,h_inv, tagsize)
            warped_binary = (warped >= 127).astype(np.uint8) * 255
            #STEP8 - Verify & Decode tag from binary image
            top_right = corners[np.argmin(np.array(corners)[:,0]**2 + np.array(corners)[:,1]**2)]
            if verify_tag(warped_binary, tagsize, scaling):
                rotations, tag_id = decode_tag(warped_binary, tagsize)
                if tag_id is not None:
                    detections.append([tag_id, corners, rotations])
                    # og_frame = draw_contours(og_frame,[255,0,0],[components])
                    og_frame = cv2.putText(og_frame,"ID: "+str(tag_id)+" ROT: "+str(rotations),top_right,cv2.FONT_HERSHEY_PLAIN,1.3,[0,0,255],2)         
    # og_frame = cv2.putText(og_frame,"FPS: "+str(1//(time.time()-start)),(10,50),cv2.FONT_HERSHEY_PLAIN,3,[0,0,0],3)
    return og_frame, detections

def warp_overlay(scene, overlay, corners, H):
    h_ol, w_ol = overlay.shape[:2]

    corners = np.array(corners)
    xmin, ymin = corners.min(axis=0).astype(int)
    xmax, ymax = corners.max(axis=0).astype(int)

    # Create grid of (x, y) inside bounding box
    xs = np.arange(xmin, xmax + 1)
    ys = np.arange(ymin, ymax + 1)
    X, Y = np.meshgrid(xs, ys)

    # Flatten grid
    Xf = X.ravel()
    Yf = Y.ravel()

    # Create homogeneous coordinates
    ones = np.ones_like(Xf)
    pts = np.stack([Xf, Yf, ones], axis=0)   # shape (3, N)

    # Apply homography
    q = H @ pts

    # Normalize
    u = q[0] / q[2]
    v = q[1] / q[2]

    ui = u.astype(np.int32)
    vi = v.astype(np.int32)

    # Valid mask
    valid = (
        (ui >= 0) & (ui < w_ol) &
        (vi >= 0) & (vi < h_ol)
    )

    # Apply mapping
    scene[Yf[valid], Xf[valid]] = overlay[vi[valid], ui[valid]]

    return scene

def task2(img, overlay, detections, tagsize=120):
    frame = img.copy()
    for detection in detections:
        [tag_id, corners, rotation] = detection
        rotated_corners = corners[rotation:] + corners[:rotation]
        try:
            h = compute_homography(rotated_corners,tagsize)
        except:
            continue
        # overlay = resize(overlay, tagsize, tagsize)
        frame = warp_overlay(frame, overlay, corners, h)
    return frame


def generate_tag(cell_size=50, tag_id=0):
    """
    Generate an AR tag image with the specified ID.
    """
    # Initialize an 8x8 black grid (0 = black)
    # The 2-cell outer border is already black by default
    grid = np.zeros((8, 8), dtype=np.uint8)
    
    # Define the internal 4x4 grid (Indices 2 to 5)
    # Row 2
    grid[2, 2] = 0
    grid[2, 3] = 255
    grid[2, 4] = 255
    grid[2, 5] = 0
    
    # Row 3
    grid[3, 2] = 255
    grid[3, 3] = 255  # ID Bit 1
    grid[3, 4] = 0  # ID Bit 2
    grid[3, 5] = 255
    
    # Row 4
    grid[4, 2] = 255
    grid[4, 3] = 255  # ID Bit 4
    grid[4, 4] = 255  # ID Bit 3
    grid[4, 5] = 255
    
    # Row 5
    grid[5, 2] = 255
    grid[5, 3] = 255
    grid[5, 4] = 255
    grid[5, 5] = 0

    # Scale the 8x8 grid to a visible image size
    tag_image = np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)
    
    cv2.imwrite(f"Tag{tag_id}.png", tag_image)

    return tag_image

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame.

    Args:
        img: The current video frame.
        obj: The loaded OBJ model.
        projection: The 3D projection matrix.
        model: The reference image representing the surface to be augmented.
        color: Whether to render in color. Defaults to False.
    """
    DEFAULT_COLOR = (0, 0, 0)
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]
            cv2.fillConvexPoly(img, imgpts, color)

    return img

if __name__ == "__main__":
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    model = np.zeros((200, 200), dtype=np.uint8)

    K = np.array([
        [1406.08415449821, 2.206797873085990, 1014.136434174160],
        [0, 1417.99930662800, 566.347754321696],
        [0, 0, 1]
    ])

    R = np.eye(3)
    t = np.array([[0],[0],[1000]])

    Rt = np.hstack((R, t))
    projection = K @ Rt

    obj = OBJ("assets/model1.obj", swapyz=True)

    img = render(img, obj, projection.astype(np.float32), model, color=False)

    cv2.imshow("Render Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()