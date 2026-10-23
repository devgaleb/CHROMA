"""CHROMA background removal. Neural segmentation for ordinary scenes; chroma key for screens."""
import cv2
import numpy as np

_SESSION = None

def alpha_for_bgr(frame, method, color, tolerance, softness, session=None):
    if method == 'AI · any background':
        try:
            from rembg import remove, new_session
        except ImportError as exc:
            raise RuntimeError('AI background removal needs rembg. Install: pip install "rembg[cpu]" (or rembg[gpu]). Restart CHROMA. The model downloads on first use.') from exc
        if session is None:
            global _SESSION
            if _SESSION is None:
                _SESSION = new_session('isnet-general-use')
            session = _SESSION
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = remove(rgb, session=session, only_mask=True)
        matte = np.asarray(result)
        if matte.ndim == 3:
            matte = matte[:, :, 0]
        if matte.shape != frame.shape[:2]:
            matte = cv2.resize(matte, (frame.shape[1], frame.shape[0]))
        return np.uint8(np.clip(matte, 0, 255))
    from chroma_pipeline import matte_key
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return np.uint8(np.clip(matte_key(rgb, color, tolerance, softness), 0, 1)*255)

def remove_frame(frame, method, color, tolerance, softness, shrink, feather, despill, min_area, bg=None):
    from app import key_frame, rgb_hex
    if method == 'Green / blue screen':
        return key_frame(frame, rgb_hex(color), tolerance, softness, int(shrink), feather, despill, int(min_area), bg)
    alpha = alpha_for_bgr(frame, method, color, tolerance, softness)
    if shrink:
        size = 2*int(shrink)+1
        alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
    if feather:
        alpha = cv2.GaussianBlur(alpha,(0,0),float(feather)*.6)
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    a = alpha.astype(np.float32)/255
    if bg is not None:
        back = cv2.cvtColor(cv2.resize(bg,(frame.shape[1],frame.shape[0])),cv2.COLOR_BGR2RGB)
        return np.uint8(np.clip(rgb*a[:,:,None]+back*(1-a[:,:,None]),0,255))
    return np.dstack((rgb,alpha))
