# Motion QC Pro

This CHROMA module estimates optical flow from the current frame back to the
previous frame and warps the previous alpha matte into current-frame coordinates.
It then compares the motion-aligned matte with the current key result.

## Outputs
- `motion_qc.json`: frame-by-frame metrics and review flags.
- `review_sheet.jpg`: original frame, grayscale matte, and checkerboard composite.

## Known limitations
Occlusions, rapid motion, screen contamination, very fine hair, and flow failures
can cause false positives or false negatives. The tool does not reconstruct hair
or certify a render. Review flagged frames at native resolution.
