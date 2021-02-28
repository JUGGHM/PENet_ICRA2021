import os
if not ("DISPLAY" in os.environ):
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

cmap = plt.cm.jet
cmap2 = plt.cm.nipy_spectral

def validcrop(img):
    ratio = 256/1216
    h = img.size()[2]
    w = img.size()[3]
    return img[:, :, h-int(ratio*w):, :]

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

def feature_colorize(feature):
    feature = (feature - np.min(feature)) / ((np.max(feature) - np.min(feature)))
    feature = 255 * cmap2(feature)[:, :, :3]
    return feature.astype('uint8')

def mask_vis(mask):
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = 255 * mask
    return mask.astype('uint8')

def merge_into_row(ele, pred, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None):
    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return depth_colorize(y)

    # if is gray, transforms to rgb
    img_list = []
    if 'rgb' in ele:
        rgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        rgb = np.transpose(rgb, (1, 2, 0))
        img_list.append(rgb)
    elif 'g' in ele:
        g = np.squeeze(ele['g'][0, ...].data.cpu().numpy())
        g = np.array(Image.fromarray(g).convert('RGB'))
        img_list.append(g)
    if 'd' in ele:
        img_list.append(preprocess_depth(ele['d'][0, ...]))
        img_list.append(preprocess_depth(pred[0, ...]))
    if extrargb is not None:
        img_list.append(preprocess_depth(extrargb[0, ...]))
    if predrgb is not None:
        predrgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        predrgb = np.transpose(predrgb, (1, 2, 0))
        #predrgb = predrgb.astype('uint8')
        img_list.append(predrgb)
    if predg is not None:
        predg = np.squeeze(predg[0, ...].data.cpu().numpy())
        predg = mask_vis(predg)
        predg = np.array(Image.fromarray(predg).convert('RGB'))
        #predg = predg.astype('uint8')
        img_list.append(predg)
    if extra is not None:
        extra = np.squeeze(extra[0, ...].data.cpu().numpy())
        extra = mask_vis(extra)
        extra = np.array(Image.fromarray(extra).convert('RGB'))
        img_list.append(extra)
    if extra2 is not None:
        extra2 = np.squeeze(extra2[0, ...].data.cpu().numpy())
        extra2 = mask_vis(extra2)
        extra2 = np.array(Image.fromarray(extra2).convert('RGB'))
        img_list.append(extra2)
    if 'gt' in ele:
        img_list.append(preprocess_depth(ele['gt'][0, ...]))

    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_image_torch(rgb, filename):
    #torch2numpy
    rgb = validcrop(rgb)
    rgb = np.squeeze(rgb[0, ...].data.cpu().numpy())
    #print(rgb.size())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb.astype('uint8')
    image_to_write = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_depth_as_uint16png(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256).astype('uint16')
    cv2.imwrite(filename, img)

def save_depth_as_uint16png_upload(img, filename):
    #from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256.0).astype('uint16')
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    imgsave.save(filename)

def save_depth_as_uint8colored(img, filename):
    #from tensor
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = depth_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_mask_as_uint8colored(img, filename, colored=True, normalized=True):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    if(normalized==False):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    if(colored==True):
        img = 255 * cmap(img)[:, :, :3]
    else:
        img = 255 * img
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def save_feature_as_uint8colored(img, filename):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = feature_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
