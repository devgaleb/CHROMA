# CHROMA | Finishing guide

## Rotoscope Lab
Open a foreground in the image editor, paint white on protected regions or choose Erase foreground for unwanted areas, then render a corrected matte. Export the PNG and name it `000000.png` for frame 0, `000001.png` for frame 1, etc. Zip frame mattes and load them under VFX > correction masks. These are per-frame manual overrides; no automated propagation of painted strokes is implied.

## Hair / fine edges
The guided filter refines alpha edges according to luminance structure in the input. It cannot reconstruct hair occluded by a solid background or repair fundamentally blurred source images.

## Color management
Install `pip install opencolorio` to enable OCIO transformations. Upload your studio `.ocio` config and provide valid named input/output spaces. OCIO conversion applies to the finished RGB pixels. The rest of the pipeline uses float32 working buffers and approximate sRGB linearization; this is not a complete ACES reference implementation.

## Precision and renders
Enable 16-bit decode for FFmpeg RGB48LE input. This can preserve 10/12/16-bit source detail through the decode stage where FFmpeg supports it, but cannot add precision to 8-bit originals. Export float EXR or ProRes 4444 as needed. Frame sequences are generated one frame at a time, with per-frame optical flow state; there is no distributed renderer or GPU farm scheduler. Large shots need adequate disk space and RAM. FFmpeg and ffprobe must be installed for precision video decoding.

## Dependencies
Core: `pip install -r requirements.txt`. Optional OCIO: `pip install opencolorio`. System binaries: `ffmpeg` and `ffprobe`.
