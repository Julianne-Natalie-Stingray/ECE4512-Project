Input an image, returns its pixelated version.
For example: 
- $1920\times 1080$ image
- Block size: $10 \times 10$ by internal calculation
- Color representative: $32$
- Returns an image of $192 \times 108$, $32$ color.
# 1	References:
- [youtu.be/7ChVezZPv64?si=6UcK7d1WsK7yOG9-](https://youtu.be/7ChVezZPv64?si=6UcK7d1WsK7yOG9-) 
- [esimov/triangle: Convert images to computer generated art using delaunay triangulation.](https://github.com/esimov/triangle)  
- [sedthh/pyxelate: Python class that generates pixel art from images](https://github.com/sedthh/pyxelate)  
- [AI 像素素材生成器｜游戏图标、平铺纹理、序列帧 - Pix Forge](https://www.mcwar.cn/#auth-panel)  
- [反向弯曲的个人空间-反向弯曲个人主页-哔哩哔哩视频](https://space.bilibili.com/16175054/lists/1911515?type=season)  
- [theamusing/perfectPixel: Refine and quantize messy AI pixel art into clean, perfect pixels.](https://github.com/theamusing/perfectPixel)  
- [PerfectPixel Processor](https://theamusing.github.io/perfectPixel_webdemo/)  
- [【保姆级教程】ai像素画，真：一键变成完美像素！_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1p3raB8EMh/?spm_id_from=0.0.search-card.all.click&vd_source=96d8f79bfcb0febea1714dbd22b6ba4c)
# 2	Difficulty
- Block size?
- Dirty color. How to pick color representatives?
- Blurred edges. How to make generation edge-aware?
- Should edge be the same size as the block (brush size), or should it be thinner? I prefer the former because this is in line with pixel drawing design (like aseprite), but how will it represent fine curves and edges that are important but are only a few pixels wide? The brush size idea implies a small block size.
- Should the pixelated version be of the same size as the original? In other words, if we pixelate the original image with block size = 10, should it becomes $\frac{1}{10}$ of its former size, or should the pixels becomes 10 times larger? I prefer the former because then the user won't have to worry about the size. Returning a significantly smaller image requires further resizing, which introduces unnessesary bias.
# 3	Project Structure:
## 3.1	Resize Image
Resize the image to a reasonable scale while preserving its ratio.
## 3.2	Identify Block Size
Use the method similar to the GitHub project [[perfectPixel_readme|perfectPixel]]: Fourier frequency peak analysis and edge detection together determines the suitable block size. But its inputs are approximately pixelated, while ours are not. Is this approach portable?
## 3.3	Color Palette Generation
Directly compress the image makes. Use KNN-cluster or other classification algorithm to extract adequate amount of representative colors. Alternatively, use AI models for color recommendation.
## 3.4	Edge Detection
Gradient method.
## 3.5	Populate Flat Area
 Find nearest color match from the palette. for each flat area, apply that color.
## 3.6	Edge Building
Build edges by taking both sides' value into account and perform interpolation. Apply unsharp sharping.
## 3.7	Final Enhancement
Perform histogram equalization.
Better: enhance the original image to make sure the output is consistent with color palette.
# 4	Project lifecycle
## 4.1	Stage 1: minimal loop
- Read and scale image
- Simple KNN palette generation
- Edge detection: Sobel
- Populate flat areas
- Edge buliding: interpolation
- Skip enhancement
# 5	MVP ver.1:
Block slicing and local color representation (skip global color representation for now) 
## 5.1	Feature:
Segment the image into blocks (size predefined) and determine the block's color and intensity based on the pixels contained. Then build the image.
## 5.2	Comment:
```
- Essentially a rough compression.
- Lacks smart block size calculation. A seam is visible at the bottom because the image isn't integer multiple of the block size.
- Dirty color. If the original image is rich in color (like the van-gogh case), the visual can be disappointing.
- Blurred edges. Unsatisfactory.
```