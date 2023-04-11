def visual_v_to_uvs():
    for i in v_to_uvs:
        if len(i) == 2:
            uv_i1, uv_i2 = list(i)[0], list(i)[1]
            uv1 = uvs[uv_i1]
            uv2 = uvs[uv_i2]
            uv1 = np.array(uv1, dtype=np.float)
            uv2 = np.array(uv2, dtype=np.float)
            uv1, uv2 = [round(i) for i in 1024 * uv1], \
                       [round(i) for i in 1024 * uv2]
            cv2.circle(img, (uv1[0], uv1[1]), 3, (0, 0, 100))
            cv2.circle(img, (uv2[0], uv2[1]), 3, (100, 5, 100))