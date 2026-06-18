Input an image, returns its pixelized version.
For example: 
- $1920\times 1080$ image
- Block size: $10 \times 10$ by internal calculation
- Color representative: $32$
- Returns an image of $192 \times 108$, $32$ color.
# 1	References:
# 2	Difficulty
- Block size?
- Dirty color. How to pick color representatives?
- Blurred edges. How to make generation edge-aware?
# 3	Project Structure:
## 3.1	Resize Image
Resize the image to a reasonable scale while preserving its ratio.
## 3.2	Identify Block Size
Use the method similar to the GitHub project [[perfectPixel_readme|perfectPixel]]: Fourier frequency peak analysis and edge detection together determines the suitable block size. But its inputs are approximately pixelized, while ours are not. Is this approach portable?
## 3.3	Color Palette Generation
Directly compress the image makes. Use KNN-cluster or other classification algorithm to extract adequate amount of representative colors. Alternatively, use AI models for color recommendation.
## 3.4	Edge Detection
Gradient method.
## 3.5	Populate Flat Area
 Find nearest color match from the palette. for each flat area
## 3.6	Edge Building
# 4	MVP ver.1:
Block slicing and local color representation (skip global color representation for now) 
## 4.1	Feature:
Segment the image into blocks (size predefined) and determine the block's color and intensity based on the pixels contained. Then build the image.
## 4.2	Comment:
```
- Essentially a rough compression.
- Lacks smart block size calculation. A seam is visible at the bottom because the image isn't integer multiple of the block size.
- Dirty color. If the original image is rich in color (like the van-gogh case), the visual can be disappointing.
- Blurred edges. Unsatisfactory.
```