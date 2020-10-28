# just concept design (port to cython or cpp later?)

import cv2
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from tqdm import tqdm

def calculateBrightness(img,w,h):
    # cr, cg, cb = red channel, green channel, blue channel
    # normalizes luminance to 0-1 (Y/width*height)
    rcoef, gcoef, bcoef = 0.241,0.691,0.068
    r,g,b = img[:,:,2], img[:,:,1], img[:,:,0]

    # need to square:
    r = rcoef*(r**2)
    g = gcoef*(g**2)
    b = bcoef*(b**2)
    _out = np.sqrt(r+g+b)/(w*h)
    return np.sum(_out)

def contrastEqualization(_img):
    tb = 125 #brightness threshold
    contrast = 1
    img = _img.copy().astype(np.float64)
    height, width = img.shape[0], img.shape[1]
    brightness = calculateBrightness(img, width, height)

    #cv2.imshow('before', _img)

    if brightness > tb:
        while brightness > tb:
            contrast -= 0.01
            img *= contrast
            brightness = calculateBrightness(img,width,height)

    return img.astype(np.uint8), brightness

def specularDetect(img, show_only_mask=False):
    # preprocessing:
    h,w = img.shape[:2]
    post_img, brightness = contrastEqualization(img.copy())
    #convert to HSV!
    hsv_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2HSV)
    specular_mask = np.ndarray((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    # compute T_v frm y=2x equation:
    # brightness from Value:
    T_v = 2*brightness
    # compute k_v value:
    kv = int(T_v/brightness)

    # Saturation threshold (T_s) is a constant (but what value? wtf):
    T_s = 170

    # Automatic threshold reconfiguration based on histogram value:
    # Get histogram values:
    histval = hsv_img[:,:,2].ravel()
    count_hv = Counter(histval)

    if count_hv[255] > ((hsv_img.shape[0]*hsv_img.shape[1])/3):
        T_s = 30
        T_v = 245

    print(T_s, T_v, kv)

    # Threshold declaration:
    for x in range(len(hsv_img)):
        for y in range(len(hsv_img[x])):
            s_x = hsv_img[x][y][1]
            v_x = hsv_img[x][y][2]

            if s_x < T_s and v_x > T_v:
                specular_mask[x][y] = 1
            else:
                specular_mask[x][y] = 0

    # Gradient post-processing:
    contours, hierarchy = cv2.findContours(specular_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    k_segments = [(w//kv)*i for i in range(kv)]
    #print('k_segments: ',k_segments)
    bbox = []

    for i,con in enumerate(contours):
        con = con.reshape(-1,2)
        #print("con length:", len(con))
        x_max, y_max = con[0]
        x_min, y_min = con[0]

        for (x,y) in con:
            x_max = max(x,x_max)
            y_max = max(y,y_max)
            x_min = min(x,x_min)
            y_min = min(y,y_min)
            #cv2.circle(hsv_img, (x, y), 1, (0, 255, 255), 2)

        # remove all the one-pixel boxes...:
        if x_max-x_min > 1 and y_max-y_min > 1:
            bbox.append((x_min, y_min, x_max, y_max))

    x = len(bbox)
    max_amount = int(math.sqrt(2*(x+10))/2)
    if max_amount < 5:
        max_amount = 5;
    if len(bbox) < max_amount:
        max_amount = len(bbox)

    area = [(i,(bb[2]-bb[0])*(bb[3]-bb[1])) for i,bb in enumerate(bbox)]
    def _sort(item):
        return item[1]
    area = sorted(area, key=_sort)
    area = area[len(area)-max_amount:-1]

    print(len(bbox))
    print("max amount: ", max_amount)

    test_img = hsv_img.copy()
    #cv2.imshow('original', cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR))
    for i,_ in area:
        bb = bbox[i]
        x_max, y_max = bb[2], bb[3]
        x_min, y_min = bb[0], bb[1]
        center_pt = (x_min+((x_max-x_min)//2),y_min+((y_max-y_min)//2))
        cv2.circle(test_img, center_pt, 1, (0, 255, 255), 2)
        cv2.rectangle(test_img, (x_min, y_min),(x_max, y_max), (100,255,255), 4)


    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5,5),np.float32)/25
    #rgb_img = cv2.filter2D(rgb_img,-1, kernel)
    if show_only_mask:
        return np.dstack([specular_mask*255]*3)
    cv2.imshow('spec', specular_mask.astype(np.float64))
    #cv2.imshow('contours', cv2.cvtColor(test_img, cv2.COLOR_HSV2BGR))
    #cv2.waitKey(-1)

    glare = readImage('flare_template.png')
    glare_r,glare_g,glare_b = glare[:,:,0]/255,glare[:,:,1]/255,glare[:,:,2]/255

    # Glare generation:
    for i,a in tqdm(area):
        bb = bbox[i]
        x_max, y_max = bb[2], bb[3]
        x_min, y_min = bb[0], bb[1]
        center_pt = (x_min+((x_max-x_min)//2),y_min+((y_max-y_min)//2))
        gen_size = random.randint(10,30)
        x_coors = [center_pt[0]-gen_size, center_pt[0]+gen_size]
        x_coors = np.clip(x_coors, 0, w)
        y_coors = [center_pt[1]-gen_size, center_pt[1]+gen_size]
        y_coors = np.clip(y_coors, 0, h)

        segment = rgb_img[y_coors[0]:y_coors[1], x_coors[0]:x_coors[1]]
        #sup = math.e**(-0.25*(math.e**(-1*(a/(img.shape[0]*img.shape[1])))))
        maskOverlay(segment, random.choice([glare_r, glare_g, glare_b]), suppress=1)

    #cv2.imshow("test2",rgb_img)
    #cv2.waitKey(-1)
    #rgb_img = gaussianBloom(rgb_img)
    return rgb_img

def pixel_hsv_to_rgb(pixel):
    h,s,v = pixel
    c = v*s
    x = c*(1-abs(((h/60)%2)-1))
    m = v-c

    if h < 60 and h >= 0:
        rp,gp,bp = c,x,0
    elif h < 120 and h >= 60:
        rp,gp,bp = x,c,0
    elif h < 180 and h >= 120:
        rp,gp,bp = 0,c,x
    elif h < 240 and h >= 180:
        rp,gp,bp = 0,x,c
    elif h < 300 and h >= 240:
        rp,gp,bp = x,0,c
    elif h < 360 and h >= 300:
        rp,gp,bp = c,0,x

    r,g,b = (rp+m)*255, (gp+m)*255, (bp+m)*255
    return [r,g,b]

def maskOverlay(img_segment, overlay_mask, suppress=1):
    # determine main color:
    hsv_seg = cv2.cvtColor(img_segment, cv2.COLOR_BGR2HSV)
    hist_hue = img_segment[:,:,0].ravel()
    counter_hue = Counter(hist_hue)
    counter_hue = counter_hue.most_common()
    color_to_add = np.array(pixel_hsv_to_rgb([counter_hue[0][0], 0.05, 1]))

    # TODO: Do a check here and resize the overlay_mask appropriately:
    # Add some padding to l/r and t/b to make sure its centered with the bounds of the img_segment size
    # Assume glare is always 1:1 size ratio!
    img_x, img_y = img_segment.shape[1], img_segment.shape[0]
    over_x, over_y = overlay_mask.shape[1], overlay_mask.shape[0]
    big_dim = min(img_x, img_y)
    overlay_mask = cv2.resize(overlay_mask, (0,0), fx=big_dim/over_x, fy=big_dim/over_y)

    if overlay_mask.shape[0] == img_x and overlay_mask.shape[1] == img_y:
        new_overlay = overlay_mask
    else:
        new_overlay = np.zeros((img_segment.shape[:2]))
        pad_to = max(img_x, img_y)
        axis_pad = 0 # 1 x 1 y
        if pad_to == img_y:
            axis_pad = 1

        amount_to_pad = (pad_to-big_dim)/2 # one-side padding; technically just a starting point for applying old overlay to new overlay
        if amount_to_pad % 2 != 0:
            amount_to_pad = math.floor(amount_to_pad)
        amount_to_pad = int(amount_to_pad)

        # fill them in:
        for x in range(big_dim):
            for y in range(big_dim):
                if axis_pad:
                    new_overlay[x+amount_to_pad][y] = overlay_mask[x][y]
                else:
                    new_overlay[x][y+amount_to_pad] = overlay_mask[x][y]

    for x in range(len(overlay_mask)):
        for y in range(len(overlay_mask[x])):
            mask = new_overlay[x][y]
            mask_trans = mask*suppress
            testing = np.array(((1-mask_trans)*img_segment[x][y])+(mask_trans*color_to_add)).astype(np.uint8)
            img_segment[x][y] = testing

    return img_segment

def readImage(path):
    img = cv2.imread(path)
    assert type(img) != type(None)
    return img

def videoEncoding(input_path, output_path, show_mask=False):
    cap = cv2.VideoCapture(input_path)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            frame = bling(frame, show_only_mask=show_mask)
            cv2.imshow('frame', frame)
            out.write(frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def gaussianBloom(img):
    gauss_img = cv2.GaussianBlur(img.copy(), (5,5), sigmaX=5, sigmaY=5).astype(np.float64)
    for x in range(len(gauss_img)):
        for y in range(len(gauss_img[x])):
            gauss_img[x][y][0] = img[x][y][0]+gauss_img[x][y][0]-img[x][y][0]*gauss_img[x][y][0]/255
            gauss_img[x][y][1] = img[x][y][1]+gauss_img[x][y][1]-img[x][y][1]*gauss_img[x][y][1]/255
            gauss_img[x][y][2] = img[x][y][2]+gauss_img[x][y][2]-img[x][y][2]*gauss_img[x][y][2]/255
    return gauss_img.astype(np.uint8)

def bling(img, show_only_mask=False):
    return specularDetect(img, show_only_mask=show_only_mask)

if __name__ == '__main__':
    videoEncoding('airplane.mp4','airplane_final_without_gaussian_blur.avi')
    #videoEncoding('boat.mp4','boat_mask.avi', show_mask=True)
    #img = readImage('airplane2.jpg')
    #cv2.imshow('test', bling(img))
    #cv2.waitKey(-1)
