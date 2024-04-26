import torch
from torch import nn
import torch.nn.functional as F

EPSILON = 1e-5

def similarity(u, v):
    cos_sim = nn.CosineSimilarity(dim=0,eps=EPSILON)
    cosine_sim = torch.clamp(cos_sim(u, v), -1., 1.)
    similarity = 1 - torch.sqrt(0.5*(1-cosine_sim) + EPSILON)
    return similarity
        
def TIK_LOSS(H1, H2, tradeoff_angle = 0.01 , tradeoff_scale = 0.005, device = 'cuda'):
    b,p = H1.shape

    A = torch.cat((torch.ones(b,1).to(device), H1), 1)
    B = torch.cat((torch.ones(b,1).to(device), H2), 1)

    cov_A = (A.t()@A) +(EPSILON*torch.eye((A.t()@A).shape[0]).to(device))
    cov_B = (B.t()@B) + (EPSILON*torch.eye((B.t()@B).shape[0]).to(device))
    
    _,L_A,_ = torch.linalg.svd(cov_A)
    _,L_B,_ = torch.linalg.svd(cov_B)
    
    #L_A = torch.lobpcg(cov_A, k=1, tol=EPSILON)[0]
    #L_B = torch.lobpcg(cov_B, k=1, tol=EPSILON)[0]

    A = torch.linalg.inv(cov_A)
    B = torch.linalg.inv(cov_B)
    
    cos = torch.dist(torch.ones((p+1)).to(device),(similarity(A,B)),p=1)/(p+1)
    return tradeoff_angle*(cos) + tradeoff_scale*torch.dist((L_A[:1]),(L_B[:1]))
    
def DARE_GRAM(H1, H2, treshold = 0.9, tradeoff_angle = 0.01 , tradeoff_scale = 0.005, device = 'cuda'): 
    b,p = H1.shape

    A = torch.cat((torch.ones(b,1).to(device), H1), 1)
    B = torch.cat((torch.ones(b,1).to(device), H2), 1)


    cov_A = (A.t()@A) 
    cov_B = (B.t()@B)

    _,L_A,_ = torch.linalg.svd(cov_A)
    _,L_B,_ = torch.linalg.svd(cov_B)

    eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
    eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()

    if(eigen_A[1]>treshold):
        T = eigen_A[1].detach()
    else:
        T = treshold
        
    index_A = torch.argwhere(eigen_A.detach()<=T)[-1]

    if(eigen_B[1]> treshold):
        T = eigen_B[1].detach()
    else:
        T = treshold

    index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
    
    k = max(index_A, index_B)[0]

    A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
    B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
    
    cos = torch.dist(torch.ones((p+1)).to(device),(similarity(A,B)),p=1)/(p+1)
    return tradeoff_angle*(cos) + tradeoff_scale*torch.dist((L_A[:k]),(L_B[:k]))/k
