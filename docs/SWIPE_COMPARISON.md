# CHROMA Swipe Comparison

The Swipe comparison tab provides a draggable original/processed preview. Select one of three bundled clips or upload your own footage. The preview uses CHROMA’s built-in chroma key and shows the removed background as a checkerboard. It is limited to four seconds at up to 640 pixels for fast iteration. The standard render tab handles longer/full-resolution exports. The preview does not contain a true alpha video channel: it composites transparency onto a checkerboard. Videos are encoded locally with OpenCV and, when FFmpeg is available, converted to H.264 for browser compatibility.

The three supplied clips are demonstration media. Confirm redistribution rights before publishing the repository or distributing its ZIP.
