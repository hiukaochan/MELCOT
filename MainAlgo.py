import torch
import torch.nn as nn
from OptimalTransportModule import *
from Cost_Functions.DNN import *
from Cost_Functions.LR import LRModel
from Cost_Functions.Tabnet import TabNetModel
from Cost_Functions.DNN import DNNModel
from sklearn.multioutput import MultiOutputRegressor
from Cost_Functions.FFTransformer import FFTransformerModel
from tqdm import tqdm
import copy
import random
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import max_error
from sklearn.preprocessing import StandardScaler

class Algorithm( OT, DNNModel, LRModel , nn.Module):
    def __init__(self, 
                 epoch: int = 10,
                 lr=1e-2,
                 seed=25):
    
        super().__init__()
        self.x = None  
        self.y = None  
        self.x_test = None
        self.y_test=None
        self.epoch = epoch
        self.lr = lr
        self.optimizer = None
        self.ot = None
        self.los = []
        self.theta_avg = None
        self.sigmoid_= True
        self.mse_lis=[]
        self.mae_lis=[]
        self.prepro = False
        self.scaler = None
        self.seed=seed
        self.mar_pred= None
        self.idx_las = None
        
    def Training(self, save_loss=False, gamma = 1e-1,delta=1e-2, mu=1e-2, loss_typ="MSE", model_typ="DNN",hidden_dim = 10,dropout_rate=0, solver = "SGD", LCOT_typ = "EOT", weight_decay =1e-5, momentum=0.9, OT_eps=0.1, OT_iter=50, OT_tol=1e-2, s=0.9, test_cur=False, d_model=64, alpha=10):
        self.idx_las = np.arange(self.x[:,:,0].flatten().shape[0])

        if model_typ == "DNN":    
            self.model = DNNModel(input_dim = (self.x[:,:,0].flatten())[self.idx_las].shape[0], hidden_dim = hidden_dim, output_dim = self.y[:,:,0].flatten().shape[0], dropout_rate=dropout_rate) 
        if model_typ == "LR":
            self.model = LRModel(input_dim = (self.x[:,:,0].flatten())[self.idx_las].shape[0],  output_dim = self.y[:,:,0].flatten().shape[0]) 
        if model_typ == "FFT":
            self.model = FFTransformerModel(n_features = self.x.shape[0],  n_sports = self.y.shape[0], n_countries=self.y.shape[1], d_model = d_model) 
        self.model.sigmoid_ne = self.sigmoid_
        
        W_ = nn.Parameter(torch.randn(self.y.shape[0], self.x.shape[0]))       
        nn.init.xavier_uniform_(W_)
        self.model.W=W_
        if solver == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=momentum, weight_decay=weight_decay)
        if solver == "ADAM":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        if solver == "ADAMW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=weight_decay
            )
        if solver == "ADADE":
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.lr)
        
        x__ = self.x.numpy()   
        dim1, dim2, n_samples = x__.shape
        X_flat = x__.transpose(2, 0, 1).reshape(n_samples, -1)
        self.scaler = StandardScaler()
        X_flat_scaled = self.scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, dim1, dim2).transpose(1, 2, 0)
        self.x = torch.from_numpy(X_scaled)
        
        if LCOT_typ == "EOT":
            self.ot = EOT(epsilon=OT_eps, n_iters= OT_iter, tol= OT_tol)
        if LCOT_typ == "EPOT":
            self.ot = EPOT(epsilon=OT_eps, n_iters= OT_iter, tol= OT_tol, s = s)
        if LCOT_typ == "OT":
            self.ot = OT()
        if LCOT_typ == "POT":
            self.ot = POT(s = s)
        pbar = tqdm(range(self.epoch))
        
        for epoch in pbar:           
            idx = torch.randperm(self.x.shape[2])  
            x_0 = self.x[:, :, idx]
            y_0 = self.y[:,:,idx]   
            for iter in range(self.y.shape[-1] ): 
                loss_per_it = []
                x_ = x_0[:, :, iter]  
                y_ = y_0[:, :, iter] 
                a = y_.sum(dim=1)     
                b = y_.sum(dim=0)
                self.optimizer.zero_grad()
                if model_typ == "FFT" :
                    C = self.model(x_).reshape(self.y.shape[0], self.y.shape[1])             
                    T = self.ot(C = C, a = a.float(), b = b.float()) 
                else:
                    C = self.model(x_.reshape(-1)[self.idx_las]).reshape(self.y.shape[0], self.y.shape[1])             
                    T = self.ot(C = C, a = a.float(), b = b.float())
                    
                if loss_typ == 'MSE':    
                    loss = ((T - y_/y_.sum()) ** 2).mean()  + gamma * ((C - y_)**2).mean()        

                    loss_per_it.append(loss.item())
                if loss_typ == "KL_di":
                    eps = 1e-8
                    T_norm = (T + eps) / (T + eps).sum()
                    y_norm = (y_ + eps) / (y_ + eps).sum()
                    loss = F.kl_div(y_norm.log(), T_norm,  reduction='batchmean') + gamma * ((C/C.sum() - y_norm)**2).mean() 
                    loss_per_it.append(loss.item())
                loss.backward()
                self.optimizer.step()
            loss_ = sum(loss_per_it)/len(loss_per_it)
            
            if save_loss:
                self.los.append(loss_)
            pbar.set_postfix(loss=loss.item())
            if test_cur:
                a = self.y_test.sum(dim=1)  
                b = self.y_test.sum(dim=0)
                mse, mae,_ = self.Testing(self.x_test, self.y_test, a ,b, print_=False, model_typ=model_typ)
                self.mse_lis.append(mse)
                self.mae_lis.append(mae)
                
            
    def Testing(self, x_test, y_test, a ,b, print_=True, model_typ="DNN", no_med=False, OT_eps=0.1, OT_iter=50, OT_tol=1e-2):
        with torch.no_grad():
            if model_typ == "Euclidean":
                x__ = self.x.numpy()   
                dim1, dim2, n_samples = x__.shape
                X_flat = x__.transpose(2, 0, 1).reshape(n_samples, -1)
                self.scaler = StandardScaler()
                X_flat_scaled = self.scaler.fit_transform(X_flat)
                self.ot = OT(epsilon=OT_eps, n_iters= OT_iter, tol= OT_tol)

            x_test = x_test.unsqueeze(-1)  
            x__ = x_test.numpy()   
            dim1, dim2, n_samples = x__.shape
            X_flat = x__.transpose(2, 0, 1).reshape(n_samples, -1)
            X_flat_scaled = self.scaler.transform(X_flat)
            X_scaled = X_flat_scaled.reshape(n_samples, dim1, dim2).transpose(1, 2, 0)
            x_test = torch.from_numpy(X_scaled)

            if model_typ=="FFT" :
                C = self.model(x_test.squeeze(-1)).view(y_test.shape)           
                T = self.ot(C = C, a = a.float(), b = b.float())
            elif model_typ == "Euclidean":
                s_1  =x_test.sum(dim=1)
                s_2 = x_test.sum(dim = 0)
                s_1 = s_1/s_1.sum()
                s_2 = s_2/s_2.sum()
                C = torch.tensor([[np.abs(s_1[i]-s_2[j]) for j in range(b.shape[0])] for i in range(a.shape[0])])          
                T = self.ot(C = C, a = a.float(), b = b.float())                
            else:   
                flat_x = (x_test.view(-1))[self.idx_las]
                
                C = self.model(flat_x).view(y_test.shape)
                T = self.ot(C=C, a=a.float(), b=b.float())
            
            if not self.prepro or no_med:
                mse = F.mse_loss(T*y_test.sum(), y_test).item()  
                mae = F.l1_loss(T*y_test.sum(), y_test).item()
                var_y = torch.var(y_test, unbiased=False)
                EVAR = 1 - torch.var(y_test - T*y_test.sum(), unbiased=False) / (var_y + 1e-8)
                ss_res = torch.sum((T*y_test.sum() - y_test) ** 2)
                ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                NRMSE = self.nrmse(T*y_test.sum(), y_test)
                RSE = self.rse(T*y_test.sum(), y_test)
                max_er = max_error((T*y_test.sum()).flatten().numpy(), y_test.flatten().numpy())
            else:
                y_test_ = torch.zeros(26, 184)
                y_test_[:, :y_test.shape[1]]=y_test
                T_ = torch.zeros(26, 184)
                T_[:, :y_test.shape[1]]=T
                mse = F.mse_loss(T_*y_test_.sum(), y_test_).item()  
                var_y = torch.var(y_test_, unbiased=False)
                EVAR = 1 - torch.var(y_test_ - T_*y_test_.sum(), unbiased=False) / (var_y + 1e-8)
                ss_res = torch.sum((T_*y_test_.sum() - y_test_) ** 2)
                ss_tot = torch.sum((y_test_ - torch.mean(y_test_)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                NRMSE = self.nrmse(T_*y_test_.sum(), y_test_)
                RSE = self.rse(T_*y_test_.sum(), y_test_)
                max_er = max_error((T_*y_test_.sum()).flatten().numpy(), (y_test_).flatten().numpy())
            if print_:
                print(f"rMSE: {np.sqrt(mse):.6f}")
                print("RSE: ", RSE)
            return mse, T
                
    def nrmse(self, pred, target):
        rmse = F.mse_loss(pred, target, reduction='mean').sqrt()
        norm = target.max() - target.min()
        return rmse / (norm + 1e-8)

    def rse(self, pred, target):
        numerator = F.mse_loss(pred, target, reduction='sum')
        denominator = F.mse_loss(target, target.mean(), reduction='sum')
        return (numerator / (denominator + 1e-8)).sqrt()
            
        
            