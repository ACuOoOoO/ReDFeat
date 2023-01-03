
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import kornia.geometry.transform as KGT
import kornia.filters as KF
import kornia.utils as KU
class MMLoss(nn.Module):
    def __init__(self, lam1=1,lam2=1, sample_n = 4096, input_size=192, sample_size=16, safe_radius_neg=7, safe_radius_pos=3, border=5, cuda=True):
        super().__init__()
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)
        self.sample_size = sample_size
        self.sample_n = sample_n
        self.safe_radius_pos = safe_radius_pos
        self.safe_radius_neg = safe_radius_neg
        self.raw_grid = KU.create_meshgrid(input_size,input_size,normalized_coordinates=False).squeeze(0)
        self.AP = nn.AvgPool2d(sample_size+1,stride=1,padding=sample_size//2)
        self.MP = nn.MaxPool2d(sample_size+1,stride=1,padding=sample_size//2)
        self.MP3 = nn.MaxPool2d(3,stride=1,padding=1)
        self.MP7 = nn.MaxPool2d(7,stride=1,padding=3)
        self.AP3 = nn.AvgPool2d(3,stride=1,padding=1)
        self.AP5 = nn.AvgPool2d(5,stride=1,padding=2)
        self.border_mask = F.pad(torch.ones([input_size-2*border,input_size-2*border]),
                                 [border,border,border,border]).unsqueeze(0).unsqueeze(0).long()
        self.running_score_sum = 1000
        self.M_mean = 1000
        self.priori1_mean = 1000
        self.priori2_mean = 1000
        self.mask1_mean = 10
        self.mask2_mean = 10
        self.mask3_mean = 10
        self.running_rep_sum = 100
        self.loss_desc_ = 0
        self.loss_peak_ = 0
        self.loss_rep_ = 0
        if cuda:
            self.AP = self.AP.cuda()
            self.MP = self.MP.cuda()
            self.MP7 = self.MP7.cuda()
            self.MP3 = self.MP3.cuda()
            self.AP3 = self.AP3.cuda()
            self.AP5 = self.AP5.cuda()
            self.raw_grid = self.raw_grid.cuda()
            #self.b
            self.border_mask = self.border_mask.cuda().long()
    def ramdom_sampler(self,feat1,score1,feat2_warped,score2_warped,good_mask):
        mask = torch.rand_like(score1*1.0)*good_mask
        t = mask.view(-1)
        thres_new = t.topk(self.sample_n,largest=True)[0][-1]
        mask = mask >= thres_new
        mask = mask.squeeze(0).detach()
        ####################################################
        feat1 = feat1[:,mask].t()
        feat2 = feat2_warped[:,mask].t()
        position = self.raw_grid[mask]
        score1 = score1[:,mask]
        score2 = score2_warped[:,mask]
        return feat1,score1,feat2,score2,position
    def compute_hard_dist(self,feat1,feat2,position):
        distx = (position[:,0].unsqueeze(1)-position[:,0].unsqueeze(0)).pow(2)
        disty = (position[:,1].unsqueeze(1)-position[:,1].unsqueeze(0)).pow(2)
        dist2 = distx + disty
        save_mask_neg = dist2<self.safe_radius_neg**2
        save_mask_pos = dist2<self.safe_radius_pos**2
        # hard mining
        simi = feat1 @ feat2.t()
        simi_a = feat1 @ feat1.t()
        simi_p = feat2 @ feat2.t()
        simi_max_pos = simi-10*(1-1.0*save_mask_pos)
        pos = torch.max(simi_max_pos,dim=0)[0].clamp(max=1-1e-5,min=-1+1e-5)
        simi = simi - 10*(simi>0.9) - 10*save_mask_neg
        neg_n = torch.max(simi,dim=0)[0].clamp(max=1-1e-5,min=-1+1e-5)
        neg_m = torch.max(simi,dim=1)[0].clamp(max=1-1e-5,min=-1+1e-5)
        simi_a = simi_a - 10*(simi_a>0.9) - 10*save_mask_neg
        neg_k = torch.max(simi_a,dim=0)[0].clamp(max=1-1e-5,min=-1+1e-5)
        simi_p = simi_p - 10*(simi_p>0.9)  - 10*save_mask_neg
        neg_j = torch.max(simi_p,dim=0)[0].clamp(max=1-1e-5,min=-1+1e-5)
        # M = ((torch.pi/2-neg_k.acos()).clamp(min=0).pow(2)/4+(torch.pi/2-neg_j.acos()).clamp(min=0).pow(2)/4+(torch.pi/2-neg_m.acos()).clamp(min=0).pow(2)/4+(torch.pi/2-neg_n.acos()).clamp(min=0).pow(2)/4+\
        #     pos.acos().pow(2)).pow(2)
        neg_cross = torch.max(neg_n,neg_m)
        # M = ((torch.pi/2-neg_k.acos()).clamp(min=0).pow(2)/3+(torch.pi/2-neg_j.acos()).clamp(min=0).pow(2)/4+(torch.pi/2-neg_m.acos()).clamp(min=0).pow(2)/4+(torch.pi/2-neg_n.acos()).clamp(min=0).pow(2)/4+\
        #     pos.acos().pow(2)).pow(2)
        M = ((torch.pi-neg_k.acos()).pow(2)/3+(torch.pi-neg_j.acos()).pow(2)/3+(torch.pi-neg_cross.acos()).pow(2)/3+\
            pos.acos().pow(2)).pow(2)
        #M = (torch.pi*3-neg_k.acos()-neg_j.acos()-neg_cross.acos()+pos.acos()*3).pow(2)
        return M
    
    def loss_desc(self):
        loss = 0
        for i in range(self.score1.shape[0]):
            # if number of samples is too samll or scores are all zeros, randomly generate a new mask
            feat1,score1,feat2,score2,position = self.ramdom_sampler(self.feat1[i],self.AP3(self.score1[i]),self.feat2_warped[i],
                                                                     self.AP3(self.score2_warped[i]),self.good_mask[i])
            assert feat1.shape[0]==feat2.shape[0]
            M = self.compute_hard_dist(feat1,feat2,position)
            score_i = score1*score2
            # generate save mask for two neighbors
            loss += (score_i.detach()*M).sum()/(self.running_score_sum+1e-5)
            self.running_score_sum = 0.99*self.running_score_sum+0.01*score_i.sum().detach()
            assert not loss.isnan()
        return loss/(i+1)
    
    def loss_rep(self):
        feat_wise_point_simi = ((self.feat1*self.feat2_warped).sum(dim=1,keepdim=True)).detach()*self.good_mask
        good_mask = F.unfold(feat_wise_point_simi,kernel_size=self.sample_size,padding=0,stride=self.sample_size//2).transpose(1,2)
        patches1 = F.unfold(self.score1*self.good_mask,kernel_size=self.sample_size,padding=0,stride=self.sample_size//2).transpose(1,2)
        patches1 = F.normalize(patches1,dim=2)
        patches2 = F.unfold(self.score2_warped*self.good_mask,kernel_size=self.sample_size,padding=0,stride=self.sample_size//2).transpose(1,2)
        patches2 = F.normalize(patches2,dim=2)
        patches_simi = F.unfold(feat_wise_point_simi,kernel_size=self.sample_size,padding=0,stride=self.sample_size//2).transpose(1,2)
        patches_simi = patches_simi.sum(dim=2,keepdim=True).clamp(min=0)/(good_mask.sum(dim=2,keepdim=True)+1)
        cosim = (patches1 * patches2).sum(dim=2,keepdim=True)
        #rep loss weighted with desciptors similairty
        loss_rep = (patches_simi*(1.0-cosim)).sum()/(self.running_rep_sum+1e-5)
        assert not loss_rep.isnan()
        self.running_rep_sum = 0.99*self.running_rep_sum + 0.01*patches_simi.sum()
        return loss_rep
    
    def compute_edge(self,im):
        #im = KF.gaussian_blur2d(im,kernel_size=(3,3),sigma=(1,1))
        # im = (im+4).clamp(min=0)
        # im = im/self.AP(im)
        edge = KF.spatial_gradient(im,order=2).abs().sum(dim=[1,2])
        edge = self.AP3(self.MP7(edge.unsqueeze(1))).detach()
        edge_min = edge.min(dim=-1,keepdim=True)[0]
        edge_min = edge_min.min(dim=-2,keepdim=True)[0]
        edge_max = edge.max(dim=-1,keepdim=True)[0]
        edge_max = edge_max.max(dim=-2,keepdim=True)[0]
        edge = (edge-edge_min)/(edge_max-edge_min)
        return edge
        
    def loss_peak(self):
        priori1 = self.compute_edge(self.im1)
        mask1 = 1-priori1/(priori1.mean()+1e-12)
        mask1_neg = self.AP((mask1<0).float())>0
        #mask_pos = mask
        mask1 = F.relu(mask1)
        priori2 = self.compute_edge(self.im2)
        mask2 = 1-priori2/(priori2.mean()+1e-12)
        mask2_neg = self.AP((mask2<0).float())>0
        mask2 = F.relu(mask2)
        score1 = self.score1*self.border_mask
        score2 = self.score2*self.border_mask
        score1_ = KF.gaussian_blur2d(score1,kernel_size=(3,3),sigma=(1,1))
        score2_ = KF.gaussian_blur2d(score2,kernel_size=(3,3),sigma=(1,1))
        loss_peak_edge = (mask1*score1.pow(2)).mean()/self.mask1_mean + (mask2*score2.pow(2)).mean()/self.mask2_mean
        # loss_peak_random = score1.pow(2).mean() + (1-self.MP(score1)).pow(2).mean() +\
        #                    score2.pow(2).mean() + (1-self.MP(score2)).pow(2).mean()
        # loss_peak_random = self.AP(score1).pow(2).mean() + (1-self.MP(score1_)).pow(2).mean() + (self.AP3(score1).pow(2)+(1-self.MP3(score1_)).pow(2)).mean()+\
        #                    self.AP(score2).pow(2).mean() + (1-self.MP(score2_)).pow(2).mean() + (self.AP3(score2).pow(2)+(1-self.MP3(score2_)).pow(2)).mean()
        loss_peak_random = self.AP(score1_).pow(2).mean() + (1-self.MP(score1_)).pow(2).mean() + (self.AP3(score1_)+1-self.MP3(score1_)).pow(2).mean()+\
                           self.AP(score2_).pow(2).mean() + (1-self.MP(score2_)).pow(2).mean() + (self.AP3(score2_)+1-self.MP3(score2_)).pow(2).mean()
        # loss_peak_random = (score1 + 1-self.MP(score1_)).pow(2).mean() + score1.pow(2).mean()+(1-self.MP3(score1_)).pow(2).mean()+\
        #                    (score2 + 1-self.MP(score2_)).pow(2).mean() + score1.pow(2).mean()+(1-self.MP3(score1_)).pow(2).mean()
        # loss_peak_random = score1.mean() + (1-self.MP(score1_)).mean()/2 + (1-self.AP3(score1_)).mean()/2+\
        #                    score2.mean() + (1-self.MP(score2_)).mean()/2 + (1-self.AP3(score2_)).mean()/2
        #loss_peak_random = score1.pow(2).mean()+(1-self.MP(score1_)).pow(2).mean()+score2.pow(2).mean()+(1-self.MP(score2_)).pow(2).mean()
        
        # loss_peak_random = score2.mean() + (1-self.MP(score2_)).mean()/2 + \
        #                     score1.mean() + (1-self.MP(score1_)).mean()/2+ \
        #                     (1-self.MP3(score1_)).mean()/2+(1-self.MP3(score2_)).mean()/2
        # loss_peak_random = (score1+1-self.MP(score1_)).mean().pow(2)/4+(score1+1-self.MP3(score1_)).pow(2).mean()/4 + \
        #                    (score2+1-self.MP(score2_)).mean().pow(2)/4+(score2+1-self.MP3(score2_)).pow(2).mean()/4
        
        # loss_kill_boarder = (score1*(1-self.border_mask)).pow(2).mean()+(score2*(1-self.border_mask)).pow(2).mean()
        loss_peak_coupled = 0
        for i in range(self.score1.shape[0]):
            # if number of samples is too samll or scores are all zeros, randomly generate a new mask
            feat1,score1,feat2,score2,position = self.ramdom_sampler(self.feat1[i],self.score1[i],self.feat2_warped[i],
                                                                     self.score2_warped[i],self.good_mask[i])
            assert feat1.shape[0]==feat2.shape[0]
            M = self.compute_hard_dist(feat1,feat2,position)
            M = (M-M.min())/(M.max()-M.min())
            mask3 = F.relu((1-M/(M.mean()+1e-12)))
            mask3 = mask3.detach()
            # generate save mask for two neighbors
            t = (mask3*(1-score1).pow(2)).mean()+(mask3*(1-score2).pow(2)).mean()
            loss_peak_coupled += t/self.mask3_mean
            self.mask3_mean = 0.99*self.mask3_mean + 0.01*mask3.mean()
        loss_peak_coupled = loss_peak_coupled/(i+1)
        self.mask1_mean = 0.99*self.mask1_mean + 0.01*mask1.mean()
        self.mask2_mean = 0.99*self.mask2_mean + 0.01*mask2.mean()
        loss_peak = (loss_peak_edge+loss_peak_random+loss_peak_coupled)
        assert not loss_peak_edge.isnan()
        assert not loss_peak_coupled.isnan()
        assert not loss_peak_random.isnan()
        assert not loss_peak.isnan()
        return loss_peak
    
    def forward(self,feat1,score1,feat2,score2,flow12=None,img1=None,img2=None):
        ones = torch.ones_like(score1)
        score1 = score1
        score2 = score2
        self.good_mask = self.generate_good_mask(ones,flow12)*self.border_mask # filter border
        self.feat2_warped = KGT.remap(feat2,flow12[..., 0], flow12[..., 1], align_corners=False,mode='nearest')
        self.score2_warped = KGT.remap(score2,flow12[..., 0], flow12[..., 1], align_corners=False,mode='bilinear')
        self.feat1 = feat1
        self.feat2 = feat2
        self.score1 = score1
        self.score2 = score2
        self.flow12 = flow12
        self.im1 = img1
        self.im2 = img2
        self.loss_desc_ = self.loss_desc()
        self.loss_peak_ = self.loss_peak()
        self.loss_rep_ = self.loss_rep()
        return self.loss_desc_+self.lam1*self.loss_peak_+self.lam2*self.loss_rep_
    
    def generate_good_mask(self,ones,flow12,n=1):
        good_mask =KGT.remap(ones*0+1,flow12[..., 0], flow12[..., 1], align_corners=False)
        t = flow12
        for i in range(n):
            good_mask = self.AP3(good_mask.float())>0.5
        # if good_mask.float().sum()<100:
        #     for i in range(good_mask.shape[0]):
        #         good_mask_i = good_mask[i]
        #         valid_points = good_mask_i>0.1
        #         if valid_points.float().sum()<100:
        #             im = TF.to_pil_image(self.im2[i].squeeze(0))
        #             im.save('error.png')
        return good_mask
        
# if __name__ == '__main__':
#     feat1 = torch.randn([2,128,192,192]).cuda()
#     feat2 = feat1 + torch.randn([2,128,192,192]).cuda()*0.3
#     feat1 = F.avg_pool2d(feat1,kernel_size=3,padding=1,stride=1)
#     feat2 = F.avg_pool2d(feat2,kernel_size=3,padding=1,stride=1)
#     feat1 = F.normalize(feat1,dim=1)
#     feat2 = F.normalize(feat2,dim=1)
#     from utils import *
#     score1 = torch.rand([2,1,192,192])
#     score2 = torch.rand([2,1,192,192])
#     #score2 = score2/score2.max()
#     score1 = score1.cuda()
#     score2 = score2.cuda()
#     aflow = KU.create_meshgrid(192,192,normalized_coordinates=False).cuda()
#     aflow = torch.cat([aflow,aflow])
#     #feat1,feat2,aflow = Random_proj(feat1,feat1) 
#     loss = MMLoss()
#     l = loss(feat1,score1,feat1,score2,aflow)
    
    