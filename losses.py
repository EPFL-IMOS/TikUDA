import torch
from torch import nn
import torch.nn.functional as F



# Function to calculate inverse of a positive definite matrix using Cholesky factorization
def inverse_cholesky(matrix):
    # Perform Cholesky factorization
    cholesky_factor = torch.linalg.cholesky(matrix)
    # Calculate inverse using Cholesky factors
    matrix_inverse = torch.cholesky_inverse(cholesky_factor)
    return matrix_inverse

# Function to calculate the largest singular value of a matrix using power iteration
def power_iteration(matrix, num_iterations=10):
    # Initialize a random vector
    vector = torch.rand(matrix.size(1), device=matrix.device)
    vector = vector / torch.norm(vector)  # Normalize the vector
    
    # Perform power iteration
    for _ in range(num_iterations):
        # Multiply by A^T * A
        Av = matrix.t() @ (matrix @ vector)
        # Normalize the resulting vector
        vector = Av / torch.norm(Av)
    
    # Calculate the largest singular value
    largest_singular_value = torch.sqrt(torch.dot(vector, matrix.t() @ (matrix @ vector)))
    return largest_singular_value




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

    cov_A = (A.t()@A) +(1e-2*torch.eye((A.t()@A).shape[0]).to(device))
    cov_B = (B.t()@B) + (1e-2*torch.eye((B.t()@B).shape[0]).to(device))
    
    _,L_A,_ = torch.linalg.svd(cov_A)
    _,L_B,_ = torch.linalg.svd(cov_B)

    A = torch.linalg.inv(cov_A)
    B = torch.linalg.inv(cov_B)
    
    cos = torch.dist(torch.ones((p+1)).to(device),(similarity(A,B)),p=1)/(p+1)
    return tradeoff_angle*(cos) + tradeoff_scale*torch.dist((L_A[:1]),(L_B[:1]))
    
def DARE_GRAM(H1, H2, treshold = 0.9, tradeoff_angle = 0.01 , tradeoff_scale = 0.002, device = 'cuda'): 
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


def Fast_TikUDA(H1, H2, tradeoff_angle = 0.01 , tradeoff_scale = 0.001 , device = 'cuda'):   

    b,p = H1.shape

    A = torch.cat((torch.ones(b,1).to(device), H1), 1)
    B = torch.cat((torch.ones(b,1).to(device), H2), 1)

    cov_A = (A.t()@A) +(1e-2*torch.eye((A.t()@A).shape[0]).to(device))
    cov_B = (B.t()@B) + (1e-2*torch.eye((B.t()@B).shape[0]).to(device))
    
    L_A = power_iteration(cov_A)
    L_B = power_iteration(cov_B)
    
    A = inverse_cholesky(cov_A)
    B = inverse_cholesky(cov_B)
    
    cos = torch.dist(torch.ones((p+1)).to(device),(similarity(A,B)),p=1)/(p+1)
    return tradeoff_angle*(cos) + tradeoff_scale*torch.dist((L_A),(L_B))
