from scipy.ndimage import center_of_mass

def re_center_image(image, size=28, xp=None):
    """
    Di chuyển trọng tâm của số về chính giữa ảnh (size x size).

    Args:
        image (np.ndarray): Ảnh đầu vào (grayscale, 2D).
        size (int): Kích thước đầu ra (mặc định: 28).
    
    Returns:
        np.ndarray: Ảnh mới đã được canh giữa.
    """
    if xp is None:
        import numpy as np
        xp = np
    
    # Tìm trọng tâm (center of mass) của ảnh
    binary = (image > 0.1).astype(float)
    cy, cx = center_of_mass(binary)
    
    # Nếu ảnh toàn 0 (không có số), thì không cần xử lý
    if xp.isnan(cx) or xp.isnan(cy):
        return image.copy()

    # Tính độ lệch cần dịch
    shift_y = int(round(size // 2 - cy))
    shift_x = int(round(size // 2 - cx))

    # Dịch ảnh bằng cách chép vùng dữ liệu
    shifted = xp.zeros_like(image)
    h, w = image.shape

    y1, y2 = max(0, shift_y), h + min(0, shift_y)
    x1, x2 = max(0, shift_x), w + min(0, shift_x)

    src_y1, src_y2 = max(0, -shift_y), h - max(0, shift_y)
    src_x1, src_x2 = max(0, -shift_x), w - max(0, shift_x)

    shifted[y1:y2, x1:x2] = image[src_y1:src_y2, src_x1:src_x2]
    return shifted
