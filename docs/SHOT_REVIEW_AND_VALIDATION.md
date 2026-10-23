# CHROMA shot review and production validation

## Visual review
Open **Shot Review & Repair**, load a video and choose a zero-based frame number. Paint foreground or erase strokes. Export the matte PNG, rename it to match the corresponding source frame, and place it in the alpha directory. Run sequence stabilization from the same panel. Stabilization uses optical flow and outputs 16-bit PNG masks. Review hair edges and scene cuts manually.

## Neural matte validation
The neural adapter requires compatible, separately obtained model weights. No newly trained weights or verified difficult-hair dataset is included. Use `chroma_quality.py --pred <predictions> --truth <reference>` with independent hand-labelled ground-truth mattes. Record MAE, MSE and RMSE, plus visual comparisons of transparent hair and motion blur. Do not claim model accuracy without those measurements.

## Color pipeline
OCIO transformations are optional and require `opencolorio` plus a studio-approved config. Verify source and destination spaces against a calibrated reference image, check float precision, EXR round-tripping, and whether alpha is premultiplied or straight. Existing export paths are not all guaranteed to be end-to-end high-bit-depth.

## Hardware acceptance
Run `python chroma_production_benchmark.py shot.mov --frames 120` on real 4K/8K footage. This measures CPU key processing only. Test actual neural inference, distributed GPU workers, color transforms and full encoding separately. Record GPU model, VRAM, driver, source codec, peak RAM, render time, and frame hashes across worker boundaries. No production-GPU results are bundled.
