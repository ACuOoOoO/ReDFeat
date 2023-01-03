from operator import pos
from turtle import forward
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from numpy.lib.type_check import imag
import kornia.geometry.transform as KGT
import kornia
import kornia.utils as KU
import kornia.filters as KF
import torch


def preprocess_image(image, preprocessing=None):
    image = image.astype(np.float32)
    image = np.transpose(image, [2, 0, 1])
    if preprocessing is None:
        pass
    elif preprocessing == 'caffe':
        # RGB -> BGR
        image = image[:: -1, :, :]
        # Zero-center by mean pixel
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean.reshape([3, 1, 1])
    elif preprocessing == 'torch':
        image /= 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    else:
        raise ValueError('Unknown preprocessing parameter.')
    return image

def RandomPosterize(x):
    #x = x*1.0/255
    th = torch.rand(1)
    x = torch.abs(x-th)
    max_ = torch.max(th,1-th)
    return x/max_

def RandomInv(x):
    if torch.rand(1)<0.5:
        return 1-x
    else:
        return x

def random_H(img=None):
    if img is not None:
        w,h = img.shape[3],img.shape[2]
    else:
        w,h = 256,256
    ratio = 0.2
    half_w = w//2
    half_h = h//2
    topleft = [
        torch.randint(0, int(ratio * half_w) + 1, size=(1, )),
        torch.randint(0, int(ratio * half_h) + 1, size=(1, ))
    ]
    topright = [
        torch.randint(w - int(ratio * half_w) - 1, w, size=(1, )),
        torch.randint(0, int(ratio * half_h) + 1, size=(1, ))
    ]
    botright = [
        torch.randint(w - int(ratio * half_w) - 1, w, size=(1, )),
        torch.randint(h - int(ratio * half_h) - 1, h, size=(1, ))
    ]
    botleft = [
        torch.randint(0, int(ratio * half_w) + 1, size=(1, )),
        torch.randint(h - int(ratio * half_h) - 1, h, size=(1, ))
    ]
    points_src = torch.FloatTensor([[
        topleft, topright, botright, botleft,
    ]])
    points_dst = torch.FloatTensor([[
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
    ]])
    #rotation
    angle = (torch.randn(1)*0.1*360).clamp(min=-10.0,max=10.0)
    #scale
    scale = (1-torch.rand(1)*0.2)*torch.ones((1,2))
    #calulate the homograph
    M_suf = KGT.get_rotation_matrix2d(torch.FloatTensor([[w/2,h/2]]),angle,scale)
    M_suf = torch.cat([M_suf,torch.FloatTensor([[[0,0,1]]])],dim=1)
    M = KGT.get_perspective_transform(points_src, points_dst)@M_suf #Homography
    return M
    
def RandomNoise(x):
    noise = torch.randn(x.shape)
    noise = noise>2
    x = (noise.float() + x).clamp(max=1)
    return x 

def random_crop_centre(img1,crop_size):
    h, w = img1.shape[2],img1.shape[3]
    right = crop_size//2 + 1
    left = w - crop_size//2 - 1
    if right >= left:
        cx = w//2
    else:
        cx = np.random.randint(right,left)
    top = crop_size//2 + 1
    bottom = h - crop_size//2 - 1
    if bottom >= top:
        cy = h//2
    else:
        cy = np.random.randint(top,bottom)
    return cx,cy
        

def Random_proj(img1,img2,crop_size=192,ratio=0.2,scale=0.2,rotation=15):
    half_cs = crop_size//2
    if len(img1.shape)<4:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    # resize if size of iamge is smaller than crop size
    min_edge = min(img1.shape[2],img1.shape[3])
    if min_edge<crop_size+3:
        img1 = KGT.resize(img1,crop_size+3)
    if min_edge<crop_size+3:
        img2 = KGT.resize(img2,crop_size+3)
    # if min_edge>2*crop_size:
    #     img1 = KGT.resize(img1,crop_size+3)
    cx,cy = random_crop_centre(img1, crop_size) #randomly generate crop centre
    w,h = img1.shape[3],img1.shape[2]
    mask = torch.ones([h,w])
    while mask.sum()>crop_size*crop_size*0.6:
    #####################################################
    #Generate Random Homography
    #Corners mapping (affine)

        
        half_w = w//2
        half_h = h//2
        topleft = [
            torch.randint(0, int(ratio * half_w) + 1, size=(1, )),
            torch.randint(0, int(ratio * half_h) + 1, size=(1, ))
        ]
        topright = [
            torch.randint(w - int(ratio * half_w) - 1, w, size=(1, )),
            torch.randint(0, int(ratio * half_h) + 1, size=(1, ))
        ]
        botright = [
            torch.randint(w - int(ratio * half_w) - 1, w, size=(1, )),
            torch.randint(h - int(ratio * half_h) - 1, h, size=(1, ))
        ]
        botleft = [
            torch.randint(0, int(ratio * half_w) + 1, size=(1, )),
            torch.randint(h - int(ratio * half_h) - 1, h, size=(1, ))
        ]
        points_src = torch.FloatTensor([[
            topleft, topright, botright, botleft,
        ]])
        points_dst = torch.FloatTensor([[
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
        ]])
        #rotation
        angle = (torch.randn(1)*360*0.1).clamp(min=-rotation,max=rotation)
        #scale
        scale = (1-torch.rand(1)*scale)*torch.ones((1,2))
        #calulate the homograph
        M_suf = KGT.get_rotation_matrix2d(torch.FloatTensor([[w/2,h/2]]),angle,scale)
        M_suf = torch.cat([M_suf,torch.FloatTensor([[[0,0,1]]])],dim=1)
        M = KGT.get_perspective_transform(points_src, points_dst)@M_suf #Homography
        M = M.to(img1.device)
        ###########################################
        
        
        
        img2_warped = KGT.warp_perspective(img2, M, (h, w), align_corners=True) #warp img2
        
        #find cropped centre on img2
        c1 = torch.FloatTensor([cx,cy,1]).unsqueeze(-1).unsqueeze(0).to(img1.device)
        c1 = torch.matmul(M,c1)
        c1 = (c1/c1[0,2,0]).long()
        if c1[0,1,0] - half_cs < 1:
            c1[0,1,0] = half_cs
        if c1[0,1,0] + half_cs > img2_warped.shape[2]-1:
            c1[0,1,0] = img2_warped.shape[2] - half_cs
        if c1[0,0,0] - half_cs < 1:
            c1[0,0,0] = half_cs
        if c1[0,0,0] + half_cs > img2_warped.shape[3]-1:
            c1[0,0,0] = img2_warped.shape[3] - half_cs
            
        #generate flow
        grid = KU.create_meshgrid(h,w,normalized_coordinates=False).to(img1.device)
        warp_grid = kornia.geometry.linalg.transform_points(M,grid) #warp flow
        warp_grid[...,0] -= c1[0,0,0]-half_cs
        warp_grid[...,1] -= c1[0,1,0]-half_cs
        warp_grid = warp_grid[:,cy-half_cs:cy+half_cs,cx-half_cs:cx+half_cs,:] #crop flow
        mask1 = warp_grid<0
        mask2 = warp_grid>=crop_size
        mask = torch.logical_or(mask1,mask2)
    warp_grid[mask] = 9e9
    
    #crop imgs
    img1_crop = img1[:,:,cy-half_cs:cy+half_cs,cx-half_cs:cx+half_cs]
    img2_crop = img2_warped[:,:,c1[0,1,0]-half_cs:c1[0,1,0]+half_cs,c1[0,0,0]-half_cs:c1[0,0,0]+half_cs]
    
    #transform output
    warp_grid = warp_grid.squeeze(0)
    img1_crop = img1_crop.squeeze(0)
    img2_crop = img2_crop.squeeze(0)
    
    return img1_crop,img2_crop,warp_grid

class pose2flow_gpu(torch.nn.Module):
    def __init__(self, size=192):
        super(pose2flow_gpu,self).__init__()
        self.raw_pos = KU.create_meshgrid(size,size,normalized_coordinates=False).cuda().reshape(-1,2).t().flip(0)
        self.size = size
        self.ones = torch.ones([1,size*size]).cuda()
    def forward(
        self,        
        depth1, intrinsics1, pose1, bbox1,
        depth2, intrinsics2, pose2, bbox2):
        
        # depth1 = depth1.view(1,1,self.size,self.size)
        # depth1 = KF.gaussian_blur2d(depth1,kernel_size=(15,15),sigma=(5,5)).view(-1)
        Z1 = depth1.view(-1)
        u1 = self.raw_pos[1, :] + bbox1[1] + .5
        v1 = self.raw_pos[0, :] + bbox1[0] + .5
        X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
        Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])
        
        XYZ1_hom = torch.cat([
            X1.view(1, -1),
            Y1.view(1, -1),
            Z1.view(1, -1),
            self.ones
        ], dim=0)
        
        XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
        XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)

        uv2_hom = torch.matmul(intrinsics2, XYZ2)
        uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)
        u2 = uv2[0, :] - bbox2[1] - .5
        v2 = uv2[1, :] - bbox2[0] - .5
        u2 = u2.long()
        v2 = v2.long()
        valid_pos2 = torch.logical_and(u2>=0,u2<depth1.shape[1])
        valid_pos2 = torch.logical_and(valid_pos2,v2>=0)
        valid_pos2 = torch.logical_and(valid_pos2,v2<depth1.shape[0])
        outlier = torch.logical_not(valid_pos2)
        v2[outlier] = 0
        u2[outlier] = 0
        annotated_depth = depth2[v2,u2]
        estimated_depth = XYZ2[2]
        annotated_depth = annotated_depth.view(1,1,self.size,self.size)
        annotated_depth = KF.gaussian_blur2d(annotated_depth,kernel_size=(15,15),sigma=(5,5))
        estimated_depth = estimated_depth.view(1,1,self.size,self.size)
        estimated_depth = KF.gaussian_blur2d(estimated_depth,kernel_size=(15,15),sigma=(5,5))
        # print(outlier.float().sum())
        # t = torch.abs(annotated_depth - estimated_depth) > 0.05
        outlier = outlier.view(1,1,self.size,self.size)
        outlier = torch.logical_or(torch.abs(annotated_depth - estimated_depth) > 0.05,outlier)
        # outlier =  KF.gaussian_blur2d(outlier.float(),kernel_size=(15,15),sigma=(5,5))>0.9
        outlier = outlier.view(-1)
        # if outlier.float().sum()>self.size*self.size*0.8:
        #     print('error')
        u2[outlier] = 1e9
        v2[outlier] = 1e9
        flow = torch.cat([u2.view(1,self.size,self.size,1),v2.view(1,self.size,self.size,1)],dim=-1)

        return flow

