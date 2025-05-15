from scipy.ndimage import center_of_mass

def re_center_image(image, size=28, xp=None):
    if xp is None:
        import numpy as np
        xp = np

    # Center of mass
    binary = (image > 0.1).astype(float)
    if xp.__name__ == 'cupy':
        binary_np = binary.get()
    else:
        binary_np = binary
    cy, cx = center_of_mass(binary_np)
    
    # Skip if image is empty
    if xp.isnan(cx) or xp.isnan(cy):
        return image.copy()

    # Calculate the shift
    shift_y = int(round(size // 2 - cy))
    shift_x = int(round(size // 2 - cx))

    shifted = xp.zeros_like(image)
    h, w = image.shape

    y1, y2 = max(0, shift_y), h + min(0, shift_y)
    x1, x2 = max(0, shift_x), w + min(0, shift_x)

    src_y1, src_y2 = max(0, -shift_y), h - max(0, shift_y)
    src_x1, src_x2 = max(0, -shift_x), w - max(0, shift_x)

    shifted[y1:y2, x1:x2] = image[src_y1:src_y2, src_x1:src_x2]
    return shifted