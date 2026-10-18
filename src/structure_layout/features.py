"""Directional clean-map features with consistent line-length filtering."""
import cv2
import numpy as np

PARAMETERS={'target_min_line_length':18,'axis_slop_px':2,'canny':[25,65],
            'direction_gradient_ratio':2,'close_length':5,'open_length':9}

def scaled_length(value, pixel_scale, *, odd=False):
    length = max(1, int(round(value*pixel_scale)))
    return length if not odd or length % 2 else length+1


def target_wall_edges(image, pixel_scale=1.):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,25,65)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    icons=(((hsv[:,:,0]<35)|(hsv[:,:,0]>170))&(hsv[:,:,1]>120)&(hsv[:,:,2]>110)).astype(np.uint8)
    icon_size=scaled_length(9,pixel_scale,odd=True)
    ignore=cv2.dilate(icons,np.ones((icon_size,icon_size),np.uint8))>0
    edges[ignore]=0
    minimum=scaled_length(PARAMETERS['target_min_line_length'],pixel_scale)
    slop=scaled_length(PARAMETERS['axis_slop_px'],pixel_scale)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=scaled_length(20,pixel_scale),
                          minLineLength=minimum,maxLineGap=scaled_length(5,pixel_scale))
    channels=np.zeros((2,*gray.shape),np.uint8)
    if lines is not None:
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            dx,dy=abs(x2-x1),abs(y2-y1)
            if dx<=slop and dy>=minimum:cv2.line(channels[0],(x1,y1),(x2,y2),1)
            elif dy<=slop and dx>=minimum:cv2.line(channels[1],(x1,y1),(x2,y2),1)
    gx=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0))
    gy=np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1))
    binary=(edges>0).astype(np.uint8)
    for channel,vertical in enumerate([True,False]):
        mask=binary*((gx>=2*gy) if vertical else (gy>=2*gx))
        close=scaled_length(PARAMETERS['close_length'],pixel_scale,odd=True)
        opening=scaled_length(PARAMETERS['open_length'],pixel_scale,odd=True)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((close,1) if vertical else (1,close),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((opening,1) if vertical else (1,opening),np.uint8))
        # Morphology must not reconnect through deliberately ignored pixels.
        mask[ignore]=0
        channels[channel]=np.maximum(channels[channel],mask)
    return channels
