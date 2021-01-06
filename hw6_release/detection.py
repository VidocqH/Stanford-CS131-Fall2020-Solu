import numpy as np
from skimage import feature, data, color, exposure, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
from scipy import signal
from scipy.ndimage import interpolation
import math

def hog_feature(image, pixel_per_cell=8):
    """
    Compute hog feature for a given image.

    Important:
    - Use the hog function provided by skimage to generate both the
      feature vector and the visualization image.
    - For the block normalization parameter, use L1!

    Args:
        image: an image with object that we want to detect.
        pixel_per_cell: number of pixels in each cell, an argument for hog descriptor.

    Returns:
        hog_feature: a vector of hog representation.
        hog_image: an image representation of hog provided by skimage.
    """
    ### YOUR CODE HERE
    pixels_per_cell = (pixel_per_cell, pixel_per_cell)
    hog_feature, hog_image = feature.hog(image, pixels_per_cell=pixels_per_cell, block_norm='L1', visualize=True)
    ### END YOUR CODE
    return (hog_feature, hog_image)

def sliding_window(image, template_feature, step_size, window_size, pixel_per_cell=8,
                   return_unresized_response=False):
    """
    A sliding window that checks each different location in the image,
    and finds which location has the highest hog score. The hog score is
    computed as the dot product between the hog feature of the sliding window
    and the hog feature of the template. It generates a response map where
    each location of the response map is a corresponding score. And you will
    need to resize the response map so that it has the same shape as the image.

    Args:
        image: np array of size (h,w). The image to apply sliding window.
        template_feature:  an array of size (m,).
            The hog representation of the object you want to find.
        step_size: int of the step size to move the window.
        window_size: pair of ints that is the height and width of the window.
        return_unresized_response: a bool to indicate returning unresized response.
            Your code can always leave this as default.
    Returns:
        max_score: float of the highest hog score.
        maxr: int of row where the max_score is found (top-left of window).
        maxc: int of column where the max_score is found (top-left of window).
        response_map: an np array of size (ceil H / step_size, ceil W / step_size).
        response_map_resized: an np array of size (h,w).
    """
    winH, winW = window_size
    H, W = image.shape
    pad_image = np.lib.pad(
        image,
        ((winH // 2,
          winH - winH // 2),
         (winW // 2,
          winW - winW // 2)),
        mode='constant')

    # Outputs: populate these!
    (max_score, maxr, maxc) = (0, 0, 0)
    response_map = np.zeros((math.ceil(H / step_size), math.ceil(W / step_size)))
    response_map_resized = np.zeros(image.shape)

    for r in range(0, H, step_size):
        for c in range(0, W, step_size):
            score = 0
            ### YOUR CODE HERE
            cur_win = pad_image[ r:r+winH, c:c+winW ]
            hog_feature = feature.hog(cur_win, pixels_per_cell=(pixel_per_cell, pixel_per_cell))
            score = hog_feature.dot(template_feature)
            if score > max_score:
                max_score = score
                maxr = r
                maxc = c
            ### END YOUR CODE
            response_map[(r) // step_size, (c) // step_size] = score

    response_map_resized = resize(response_map, image.shape, mode='constant')

    if return_unresized_response:
        return (max_score, maxr, maxc, response_map_resized, response_map)
    else:
        return (max_score, maxr, maxc, response_map_resized)


def pyramid(image, scale=0.9, min_size=(200, 100)):
    """
    Generate image pyramid using the given image and scale.
    Reducing the size of the image until either the height or
    width is below the minimum limit. In the ith iteration,
    the image is resized to scale^i of the original image.

    This function is mostly completed for you -- only a termination
    condition is needed.

    Args:
        image: np array of (h,w), an image to scale.
        scale: float of how much to rescale the image each time.
        min_size: pair of ints showing the minimum height and width.

    Returns:
        images: list containing pair of
            (the current scale of the image, resized image).
    """
    images = []

    # Yield the original image
    current_scale = 1.0
    images.append((current_scale, image))

    while True:
        # Use "break" to exit this loop when termination conditions are met.
        ### YOUR CODE HERE
        H, W = image.shape
        if H < min_size[0] and H < min_size[1]:
            break
        ### END YOUR CODE

        # Compute the new dimensions of the image and resize it
        current_scale *= scale
        image = rescale(image, scale, mode='constant')

        # Yield the next image in the pyramid
        images.append((current_scale, image))

    return images


def pyramid_score(image, template_feature, shape, step_size=20,
                  scale=0.9, pixel_per_cell=8):
    """
    Calculate the maximum score found in the image pyramid using sliding window.

    Args:
        image: np array of (h,w).
        template_feature: the hog representation of the object you want to detect.
        shape: shape of window you want to use for the sliding_window.

    Returns:
        max_score: float of the highest hog score.
        maxr: int of row where the max_score is found.
        maxc: int of column where the max_score is found.
        max_scale: float of scale when the max_score is found.
        max_response_map: np array of the response map when max_score is found.
    """
    max_score = 0
    maxr = 0
    maxc = 0
    max_scale = 1.0
    max_response_map = np.zeros(image.shape)

    images = pyramid(image, scale)
    ### YOUR CODE HERE
    for each_image in images:
        res = sliding_window(each_image[1], template_feature, \
            step_size=step_size, window_size=shape, pixel_per_cell=pixel_per_cell)
        if res[0] > max_score:
            max_score = res[0]
            maxr = res[1]
            maxc = res[2]
            max_response_map = res[3]
            max_scale = each_image[0]
    ### END YOUR CODE
    return max_score, maxr, maxc, max_scale, max_response_map


def compute_displacement(part_centers, face_shape):
    """
    Calculate the mu and sigma for each part. d is the array where each row
    is the main center (face center) minus the part center. Since in our
    dataset, the face is the full image, face center could be computed by
    finding the center of the image. Vector mu is computed by taking an average
    from the rows of d. And sigma is the standard deviation among the rows.

    Hint: d is the array of distances between part centers and the face center,
    such that d[i,0] is the face's center-x coordinate minus the ith part's x
    coordinate, and similarly for y coordinates.

    Hint: you should leave mu as floats instead of rounding to integers,
    because our next step of applying the shift in shift_heatmap will
    interpolate the shift, which is valid for float shifts.

    Args:
        part_centers: np array of (n,2) containing centers
            of one part in each image.
        face_shape: np array of (h,w) that indicates the shape of a face.
    Returns:
        mu: (2,) vector.
        sigma: (2,) vector.

    """
    d = np.zeros((part_centers.shape[0], 2))
    ### YOUR CODE HERE
    face_centers = np.array([face_shape[0] / 2, face_shape[1] / 2], dtype=np.int32)
    d = part_centers - face_centers
    mu = np.abs(np.mean(d, axis=0))
    sigma = np.std(d, axis=0)
    ### END YOUR CODE
    return mu, sigma


def shift_heatmap(heatmap, mu):
    """
    First normalize the heatmap to make sure that all the values
    are not larger than 1. This kind of normalization can be achieved via
    dividing by the maximum value of the heatmap.

    Then shift the heatmap based on the vector mu.

    Hint: use the interpolation.shift function provided by scipy.ndimage.

    Note: the arguments are copied to ensure your code doesn't alter them.
    The copies are the same type and shape as the originals.

    Args:
        heatmap: np array of (h,w).
        mu: vector array of (1,2).
    Returns:
        new_heatmap: np array of (h,w).
    """
    ### YOUR CODE HERE
    heatmap = np.copy(heatmap)
    # Normalize heatmap
    heatmap /= np.max(heatmap, axis=None)
    # Shift heatmap
    new_heatmap = interpolation.shift(heatmap, mu)
    ### END YOUR CODE
    return new_heatmap


def gaussian_heatmap(heatmap_face, heatmaps, sigmas):
    """
    Apply gaussian filter with the given sigmas to the corresponding heatmaps.
    Then add the filtered heatmaps together with the face heatmap.
    Find the index where the maximum value in the heatmap is found.

    Hint: use gaussian function provided by skimage.

    Note: the arguments are copied to ensure your code doesn't alter them.
    The copies are the same type and shape as the originals.

    Args:
        heatmap_face: np array of (h,w), corresponding to the face heatmap.
        heatmaps: list of [np array of (h,w)], corresponding to the parts heatmaps.
        sigmas: list of [np array of (2,)], corresponding to the parts sigmas.
    Return:
        heatmap: np array of (h,w), corresponding to sum of gaussian-filtered heatmaps.
        maxr: int of row where the heatmap maximum is found.
        maxc: int of column where the heatmap maximum is found.
    """
    heatmap_face = np.copy(heatmap_face)
    heatmaps = list(np.copy(heatmaps))
    sigmas = list(np.copy(sigmas))
    ### YOUR CODE HERE
    heatmap = heatmaps[0]
    maxVal = 0
    maxr = 0
    maxc = 0
    H, W = heatmap.shape
    for i in range(len(heatmaps)):
        # Apply gaussian filter
        heatmaps[i] = gaussian(heatmaps[i], sigma=sigmas[i])
        # Add filtered heatmaps with face heatmap
        heatmaps[i] += heatmap_face
        idx = np.argmax(heatmaps[i], axis=None)
        r = int(idx / W)
        c = idx - (r * W)
        if heatmaps[i][r, c] > maxVal:
            maxVal = heatmaps[i][r, c]
            maxr = r
            maxc = c
            heatmap = heatmaps[i]
    ### END YOUR CODE
    return heatmap, maxr, maxc


def detect_multiple(image, response_map):
    """
    Extra credit
    """
    detected_faces = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return detected_faces

