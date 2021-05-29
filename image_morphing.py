import cv2
import dlib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

import os
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import get_name_from_file


landmark_dict = {
    "left_eye": np.arange(36, 42),
    "left_eyebrow": np.arange(17, 22),
    "right_eye": np.arange(42, 48),
    "right_eyebrow": np.arange(22, 27),
    "nose": np.arange(31, 36),
    "nose_bridge": np.arange(27, 31),
    "lips_inner": np.arange(60, 68),
    "lips_outer": np.arange(48, 60),
    "face": np.arange(0, 17),
}

def get_face_outline_coordinates(landmarks):
    facial_landmark_ids = [
        *landmark_dict["face"],
        *landmark_dict["right_eyebrow"][::-1],
        *landmark_dict["left_eyebrow"][::-1]
    ]
    return landmarks[facial_landmark_ids]


def get_landmark_lines_fixed():
    lines = []
    for key, pts in landmark_dict.items():
        coords = np.stack([pts[:-1], pts[1:]]).T
        lines.append(coords)
    lines = np.concatenate(lines)
    return lines

def get_landmark_lines_delaunay(landmarks):
    tri = Delaunay(landmarks).simplices
    unique_lines = set()
    for pts in tri:
        p1, p2, p3 = list(sorted(pts))
        unique_lines.add((p1, p2))
        unique_lines.add((p2, p3))
        unique_lines.add((p1, p3))
    lines = []
    for x, y in unique_lines:
        lines.append([x, y])
    lines = np.array(lines)
    return lines


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return np.array(coords)

def draw_landmarks(img, lines_start, lines_end, landmarks, path):
    img = Image.fromarray(img)
    canvas = ImageDraw.Draw(img)
    # draw landmark points
    for x, y in landmarks:
        r = 2
        canvas.ellipse([(x-r, y-r), (x+r, y+r)], fill=(255, 0, 0, 255))
    # draw feature lines
    for (x1, y1), (x2, y2) in zip(lines_start, lines_end):
        canvas.line([(x1, y1), (x2, y2)], fill=(0, 0, 255, 255))
    img.save(path)
    return

def crop_resize_from_feature_points(path, detector, predictor, size=256):
    gray_img = np.array(Image.open(path).convert("L"))
    color_img = np.array(Image.open(path).convert("RGB"))

    h, w = gray_img.shape[0], gray_img.shape[1]

    rect = detector(gray_img, 1)[0]
    landmarks = predictor(gray_img, rect)
    landmarks = shape_to_np(landmarks) # pixel coordinates (x, y)

    padding = 10
    max_x, min_x = np.max(landmarks[:, 0]), np.min(landmarks[:, 0])
    max_y, min_y = np.max(landmarks[:, 1]), np.min(landmarks[:, 1])

    cropped_gray_img = gray_img[
        max(0, min_y - padding):min(max_y + padding, h), 
        max(0, min_x - padding):min(max_x + padding, w)
    ]
    cropped_color_img = color_img[
        max(0, min_y - padding):min(max_y + padding, h), 
        max(0, min_x - padding):min(max_x + padding, w)
    ]

    gray_resized = cv2.resize(cropped_gray_img, dsize=(size, size))
    color_resized = cv2.resize(cropped_color_img, dsize=(size, size))
    return gray_resized, color_resized

def get_line_start_and_end(img, detector, predictor, use_delaunay=False):
    rect = detector(img, 1)[0]
    landmarks = predictor(img, rect)
    landmarks = shape_to_np(landmarks) # pixel coordinates (x, y)
    if use_delaunay:
        corners = np.array([
            [0, 0],
            [0, image_size],
            [image_size, 0],
            [image_size, image_size]
        ])
        landmarks = np.concatenate([landmarks, corners])
        lines = get_landmark_lines_delaunay(landmarks) # line id (start, end)
    else:
        lines = get_landmark_lines_fixed()
    P = landmarks[lines[:, 0]] # P, line start
    Q = landmarks[lines[:, 1]] # Q, line end
    return P.astype(np.float64), Q.astype(np.float64), landmarks, lines

def perpendicular_vector(v):
    v_length = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
    v_homo = np.pad(v, ((0, 0), (0, 1)), mode="constant") # pad to R3, pad zeros
    z_axis = np.zeros(v_homo.shape)
    z_axis[:, -1] = 1
    p = np.cross(v_homo, z_axis)
    p = p[:, :-1] # ignore z axis
    p_length = np.sqrt(np.sum(p**2, axis=1, keepdims=True))
    p = p / (p_length + 1e-8) # now sum = 1
    p *= v_length
    return p

def warp_from_source(img_s, P_s, Q_s, P_d, Q_d):
    assert img_s.shape[0] == img_s.shape[1]
    eps = 1e-8

    perp_d = perpendicular_vector(Q_d - P_d)
    perp_s = perpendicular_vector(Q_s - P_s)
    dest_line_vec = Q_d - P_d
    source_line_vec = Q_s - P_s

    image_size = img_s.shape[0]
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
    X_d = np.dstack([x, y])
    X_d = X_d.reshape(-1, 1, 2)
    to_p_vec = X_d - P_d
    to_q_vec = X_d - Q_d
    u = np.sum(to_p_vec * dest_line_vec, axis=-1) / (np.sum(dest_line_vec**2, axis=1) + eps)
    v = np.sum(to_p_vec * perp_d, axis=-1) / (np.sqrt(np.sum(dest_line_vec**2, axis=1)) + eps)

    X_s = np.expand_dims(P_s, 0) + \
        np.expand_dims(u, -1) * np.expand_dims(source_line_vec, 0) + \
        np.expand_dims(v, -1) * np.expand_dims(perp_s, 0) / (np.sqrt(np.sum(source_line_vec**2, axis=1)).reshape(1, -1, 1) + eps)
    D = X_s - X_d
    to_p_mask = (u < 0).astype(np.float64)
    to_q_mask = (u > 1).astype(np.float64)
    to_line_mask = np.ones(to_p_mask.shape) - to_p_mask - to_q_mask

    to_p_dist = np.sqrt(np.sum(to_p_vec**2, axis=-1))
    to_q_dist = np.sqrt(np.sum(to_q_vec**2, axis=-1))
    to_line_dist = np.abs(v)
    dist = to_p_dist * to_p_mask + to_q_dist * to_q_mask + to_line_dist * to_line_mask
    dest_line_length = np.sqrt(np.sum(dest_line_vec**2, axis=-1))
    weight = (dest_line_length**p) / (((a + dist))**b + eps)
    weighted_D = np.sum(D * np.expand_dims(weight, -1), axis=1) / (np.sum(weight, -1, keepdims=True) + eps)

    X_d = X_d.squeeze()
    X_s = X_d + weighted_D
    X_s_ij = X_s[:, ::-1]

    if len(img_s.shape) == 2:
        warped = map_coordinates(img_s, X_s_ij.T, mode="nearest")
    else:
        warped = np.zeros((image_size*image_size, img_s.shape[2]))
        for i in range(img_s.shape[2]):
            warped[:, i] = map_coordinates(img_s[:, :, i], X_s_ij.T, mode="nearest")
    warped = warped.reshape(image_size, image_size, -1).squeeze()
    return warped.astype(np.uint8)

def get_intermidate_lines(P_1, Q_1, P_2, Q_2, alpha = 0.5):
    P = P_1 * alpha + P_2 * (1 - alpha)
    Q = Q_1 * alpha + Q_2 * (1 - alpha)
    return P, Q

def get_intermediate_face_outline(face1, face2, alpha = 0.5):
    return face1 * alpha + face2 * (1 - alpha)

def get_face_mask(face_outline, image_size):
    face_x = list(face_outline[:, 0])
    face_y = list(face_outline[:, 1])
    mask = Image.new("RGB", (image_size, image_size))
    canvas = ImageDraw.Draw(mask)
    canvas.polygon(list(zip(face_x, face_y)), fill = (255, 255, 255))
    mask = mask.filter(ImageFilter.GaussianBlur(10)) # soften mask
    mask = np.array(mask) * 1.0 / 255
    return mask

def compute_bokeh_image(face, mask):
    blurred = Image.fromarray(face).filter(ImageFilter.GaussianBlur(2))
    enhancer = ImageEnhance.Brightness(blurred)
    blurred = enhancer.enhance(0.6)
    blurred = np.array(blurred)
    res = face * mask + blurred * (1 - mask)
    res = res.astype(np.uint8)
    return res

def warp_and_merge(img_1, P_1, Q_1, img_2, P_2, Q_2, alpha, face_mask = None):
    start = time()
    P_inter, Q_inter = get_intermidate_lines(P_1, Q_1, P_2, Q_2, alpha)
    warped_1 = warp_from_source(img_1, P_1, Q_1, P_inter, Q_inter)
    warped_2 = warp_from_source(img_2, P_2, Q_2, P_inter, Q_inter)
    merged = warped_1 * alpha + warped_2 * (1 - alpha)
    merged = merged.astype(np.uint8)

    if face_mask is not None:
        merged = compute_bokeh_image(merged, face_mask)
    return merged

def morph_sequence(
    image_path_1, image_path_2, steps = 10, use_face_mask = False, use_delaunay = False
):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img_1, color_img_1 = crop_resize_from_feature_points(image_path_1, detector, predictor, image_size)
    img_2, color_img_2 = crop_resize_from_feature_points(image_path_2, detector, predictor, image_size)
    P_1, Q_1, landmarks_1, lines_1 = get_line_start_and_end(img_1, detector, predictor, use_delaunay)
    P_2, Q_2, landmarks_2, _  = get_line_start_and_end(img_2, detector, predictor, use_delaunay)
    if use_delaunay:
        P_2 = landmarks_2[lines_1[:, 0]]
        Q_2 = landmarks_2[lines_1[:, 1]]
    print("Num lines:", len(lines_1))

    face1_outline = get_face_outline_coordinates(landmarks_1)
    face2_outline = get_face_outline_coordinates(landmarks_2)

    warp_sequence = []
    for alpha in tqdm(np.linspace(0, 1.0, steps)):
        if use_face_mask:
            face_mask = get_face_mask(
                get_intermediate_face_outline(face1_outline, face2_outline, alpha), img_1.shape[0])
        else:
            face_mask = None
        merged = warp_and_merge(color_img_1, P_1, Q_1, color_img_2, P_2, Q_2, alpha, face_mask)
        warp_sequence.append(Image.fromarray(merged))
    return warp_sequence

def naive_morph(img1, img2, alpha):
    img = img1 * alpha + img2 * (1 - alpha)
    img = img.astype(np.uint8)
    return Image.fromarray(img)


def morph_image(image_path_1, image_path_2, alpha, use_face_mask=False, use_delaunay=False):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img_1, color_img_1 = crop_resize_from_feature_points(image_path_1, detector, predictor, image_size)
    img_2, color_img_2 = crop_resize_from_feature_points(image_path_2, detector, predictor, image_size)
    P_1, Q_1, landmarks_1, lines_1 = get_line_start_and_end(img_1, detector, predictor, use_delaunay)
    P_2, Q_2, landmarks_2, _  = get_line_start_and_end(img_2, detector, predictor, use_delaunay)
    if use_delaunay:
        P_2 = landmarks_2[lines_1[:, 0]]
        Q_2 = landmarks_2[lines_1[:, 1]]
    print("Num lines:", len(lines_1))
    
    face1_outline = get_face_outline_coordinates(landmarks_1)
    face2_outline = get_face_outline_coordinates(landmarks_2)
    draw_landmarks(color_img_1, P_1, Q_1, landmarks_1, "results/landmarks.png")
    draw_landmarks(color_img_2, P_2, Q_2, landmarks_2, "results/landmarks2.png")

    if use_face_mask:
        face_mask = get_face_mask(get_intermediate_face_outline(
            face1_outline, face2_outline, alpha), image_size)
    else:
        face_mask = None

    merged = warp_and_merge(color_img_1, P_1, Q_1, color_img_2, P_2, Q_2, alpha, face_mask)
    return Image.fromarray(merged)

def get_random_morph():
    face_ids = np.random.choice(np.arange(len(image_paths)), 2, replace=False)
    image_path_1 = os.path.join(image_dir, image_paths[face_ids[0]])
    image_path_2 = os.path.join(image_dir, image_paths[face_ids[1]])
    name1 = get_name_from_file(image_paths[face_ids[0]])
    name2 = get_name_from_file(image_paths[face_ids[1]])
    return morph_image(image_path_1, image_path_2, 0.5, True), name1, name2

def generate_morph_video(num_images, filename):
    warp_seq = []
    image_files = list(np.random.choice(image_paths, num_images, replace=False))
    image_files.append(image_files[0]) # loop to first image
    print(image_files)
    for i in range(len(image_files) - 1):
        image_path_1 = os.path.join(image_dir, image_files[i])
        image_path_2 = os.path.join(image_dir, image_files[i + 1])
        warp_seq.extend(morph_sequence(image_path_2, image_path_1, 10, True, False))
    
    warp_seq[0].save(filename, 
        save_all=True, append_images=warp_seq[1:], duration=5*num_images, loop=0)

image_size = 256
b = 2
p = 0.5
a = 1
alpha = 0.5

image_dir = "images"
predictor_path = "pretrained/shape_predictor_68_face_landmarks.dat"
image_paths = os.listdir(image_dir)
np.random.shuffle(image_paths)

if __name__ == "__main__":
    start = time()
    img = morph_image("images/brad_pitt.jpg", "images/angelina_jolie.jpg", 0.5, True, True)
    print("Spent: ", time() - start)
    img.save("results/warped_delaunay.png".format(a, b, p))
    #seq = morph_sequence("images/brad_pitt.jpg", "images/angelina_jolie.jpg", 10, True, True)
    #seq[0].save("results/brad2jolie_delaunay.gif", save_all=True, append_images=seq[1:], duration=5, loop=0)