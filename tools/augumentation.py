import os
import sys
import time
import numpy as np
import random
import copy 
import cv2

def scale_move_pad_box(row, col, scale, dst, dst_label, img_size, bbox, img_format, debug):
    x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3] 
    l = int(w*scale)
    x1 = x - random.randint(0,l-w)
    y1 = y - random.randint(0,l-h)
    x2 = x1 + l
    y2 = y1 + l
    pad = False
    p_x = 0
    p_y = 0
    p_w = 0
    p_h = 0
    if x1 < 0:
        p_x = -x1
        x1 = 0
        pad = True
    if y1 < 0:
        p_y = -y1
        y1 = 0
        pad = True
    if x2 > col:
        p_w = x2 - col
        x2 = col
        pad = True
    if y2 > row:
        p_h = y2 - row
        y2 = row
        pad = True

    if debug:
        for i in range(len(dst_label)):
            cv2.circle(dst,tuple(dst_label[i].reshape(2,1)),2,(0,255,0),-1)
        cv2.rectangle(dst,(x1,y1),(x2,y2),(255,0,0),2)

    #print "%d %d %d %d %d %d"%(x1,y1,x2,y2,col,row)
    box_img = dst[y1:y2,x1:x2]
    if pad:
        if img_format == 'GRAY' :
            box_img = np.lib.pad(box_img, ((p_y,p_h),(p_x,p_w)), 'constant')
        else:
            box_img = np.lib.pad(box_img, ((p_y,p_h),(p_x,p_w),(0,0)), 'constant')
    box_img = cv2.resize(box_img,(img_size,img_size))

    if debug:
        cv2.imwrite('./test.jpg',box_img)
        cv2.imwrite('./dst.jpg',dst)
    dst_label = ((dst_label-(x1-p_x,y1-p_y))/(l, l)).flatten()
    return box_img, dst_label

def get_rotation_center(x, y, w, h):
    return x+w/2, y+h/2

def mirror_v3(img_t,a,col):
    rimg = cv2.flip(img_t,1)
    a[:,0] = col - a[:,0]
    mirror_label = np.concatenate((a[1,:], a[0,:], a[2,:], a[4,:], a[3,:]),axis=0)
    return rimg, mirror_label 

def get_squared_bbox(x, y, w, h):
    center_x = x + w/2
    center_y = y + h/2

    l = np.amin([w, h])
    t_x = center_x-l/2 if center_x-l/2>=0 else 0
    t_y = center_y-l/2 if center_y-l/2>=0 else 0

    return t_x, t_y, l, l

def pts_augumentation(img_t, pts_t, bbox, img_size, num_pts, debug=False):
    height_t, width_t = img_t.shape[:2]
    img_t = np.reshape(img_t, [height_t, width_t, 3]) 
    pts_t = np.reshape(pts_t, [num_pts, 2])
    height_t, width_t, nChannels = img_t.shape

    x, y, w, h = get_squared_bbox(bbox[0], bbox[1], bbox[2], bbox[3])

    if np.random.rand() > 0.5:
        img_t, pts_t = mirror_v3(img_t, copy.deepcopy(pts_t), width_t)
        x = width_t-x-w

    pts_t = np.reshape(pts_t, [num_pts, 2])

    # rotation
    c_x, c_y = get_rotation_center(x, y, w, h)
    angle = random.randint(-10, 10)
    #angle = random.randint(-180, 180)
    M = cv2.getRotationMatrix2D(tuple((c_x, c_y)), angle, 1)
    img_t = cv2.warpAffine(img_t, M, (width_t, height_t))
    b = np.ones((pts_t.shape[0], 1))
    d = np.concatenate((pts_t, b), axis=1)
    pts_t = np.dot(d, np.transpose(M))

    height_t, width_t, nChannels = img_t.shape

    # scale and translation
    scale = random.uniform(1.0, 1.2) 
    box_img, dst_pts = scale_move_pad_box(height_t, width_t, scale, img_t, pts_t, img_size, [x, y, w, h], 'RGB', False)

    # visualizing
    if debug:
        for iter in range(0, num_pts):
            pts = np.reshape(dst_pts, [num_pts, 2])
            cv2.circle(box_img, tuple((pts[iter, :]*img_size).astype(int)), 2, (0,255,0), -1)
            cv2.putText(box_img,str(iter), tuple((pts[iter, :]*img_size).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imwrite('test_%f.jpg'%time.time(), box_img)

    return box_img, dst_pts.astype(np.float)

