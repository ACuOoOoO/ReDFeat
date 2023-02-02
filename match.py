import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from lib.model import MMNet
import cv2
import scipy.io as scio
from copy import deepcopy
import time
from PIL import Image
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    
def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    model = MMNet()
    weights = checkpoint['model']
    model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return model.eval()


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.6):
        super(NonMaxSuppression,self).__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rep_thr = rep_thr
        
    def forward(self, repeatability):
        #repeatability = repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        border_mask = maxima*0
        border_mask[:,:,10:-10,10:-10]=1
        maxima = maxima*border_mask
        print(maxima.sum())
        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, image_type,
                        scale_f=2**0.25, min_scale=0.0, 
                        max_scale=1, min_size=256, 
                        max_size=1024, verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")

            with torch.no_grad():
                if image_type == '1':
                    descriptors, repeatability = net.forward1(img)
                elif image_type == '2':
                    descriptors, repeatability = net.forward2(img)

            mask = repeatability*0
            mask[:,:,args.border:-args.border,args.border:-args.border] = 1
            repeatability=repeatability*mask
            y,x = detector(repeatability) # nms
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]
            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            #S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    #S = torch.cat(S) # scale
    scores = torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--num_features", type=int, default=4096, help='Number of features')
    parser.add_argument("--model", type=str, default='Pretrained/VIS_IR.pth', help='model path')
    parser.add_argument("--img1_path", type=str, default='img1.JPG', help='model path')
    parser.add_argument("--img2_path", type=str, default='img2.JPG', help='model path')
    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1000)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--border", type=float, default=5) 
    parser.add_argument("--reliability-thr", type=float, default=0.5)
    parser.add_argument("--repeatability-thr", type=float, default=0.4)

    parser.add_argument("--gpu", type=int, default=0, help='use -1 for CPU')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    net = load_network(args.model)
    net = net.cuda()
    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr = args.reliability_thr, 
        rep_thr = args.repeatability_thr)

    img1 = Image.open(args.img1_path).convert('RGB')
    W, H = img1.size
    img = TF.to_tensor(img1).unsqueeze(0)
    img = (img-img.mean(dim=[-1,-2],keepdim=True))/img.std(dim=[-1,-2],keepdim=True)
    img = img.cuda()
    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector, '1',
        scale_f   = args.scale_f, 
        min_scale = args.min_scale, 
        max_scale = args.max_scale,
        min_size  = args.min_size, 
        max_size  = args.max_size, 
        verbose = True)
    if len(scores)<args.num_features:
        idxs = scores.topk(len(scores))[1]
    else:
        idxs = scores.topk(args.num_features)[1]
    kp1 = xys[idxs].cpu().numpy()
    desc1 = desc[idxs].cpu().numpy()

    img2 = Image.open(args.img2_path).convert('RGB')
    W, H = img2.size
    img = TF.to_tensor(img2).unsqueeze(0)
    img = (img-img.mean(dim=[-1,-2],keepdim=True))/img.std(dim=[-1,-2],keepdim=True)
    img = img.cuda()
    
    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector, '2',
        scale_f   = args.scale_f, 
        min_scale = args.min_scale, 
        max_scale = args.max_scale,
        min_size  = args.min_size, 
        max_size  = args.max_size, 
        verbose = True)
    if len(scores)<args.num_features:
        idxs = scores.topk(len(scores))[1]
    else:
        idxs = scores.topk(args.num_features)[1]
    kp2 = xys[idxs].cpu().numpy()
    desc2 = desc[idxs].cpu().numpy()

    #match
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)
    src_pts = np.float32([ kp1[m.queryIdx] for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx] for m in good ]).reshape(-1,1,2)

    E, mask = cv2.findEssentialMat(
        src_pts, dst_pts, np.eye(3), threshold=5.0, prob=0.9999,
        method=cv2.RANSAC)
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    kp1 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp1]
    kp2 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp2]
    img3 = cv2.drawMatches(np.array(img1),kp1,np.array(img2),kp2,good,None,**draw_params)
    Image.fromarray(img3).save('test.png')