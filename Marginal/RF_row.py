import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

class RF_Mar:
    def __init__(self, random_state: int = 291, n_estimator=6,**rf_params):
        self.rf_params = rf_params
        self.random_state = random_state
        self.model = None
        self.scaler = None 
        self.n_estimator = n_estimator

    def build_model(self):
        base = RandomForestRegressor(random_state=self.random_state,n_estimators=self.n_estimator, **self.rf_params)
        self.model = MultiOutputRegressor(base)
        return self.model

    def train(self, x_train: torch.Tensor, y_train: torch.Tensor):
        X_np = x_train.cpu().detach().numpy()
        n_feats, n_countries, n_years = X_np.shape

        X = X_np.transpose(2,0,1).reshape(n_years, -1)

        # standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        Y_np = y_train.cpu().detach().numpy()
        Y = Y_np.transpose(2,0,1)

        Y_sum = Y.sum(axis=1)

        # build & fit
        if self.model is None:
            self.build_model()
        self.model.fit(X_scaled, Y_sum)

    def test(self, x_test: torch.Tensor, y_test: torch.Tensor):
        X_np = x_test.cpu().detach().numpy()        
        X_flat = X_np.reshape(1, -1)                
        X_scaled = self.scaler.transform(X_flat)   

        preds = self.model.predict(X_scaled)[0]     
        T = torch.from_numpy(preds)                 

        y_np = y_test.cpu().detach().numpy()        
        y_true = torch.from_numpy(y_np.sum(axis=0)) 

        mse   = F.mse_loss(T, y_true).item()
        var_y = torch.var(y_true, unbiased=False)
        EVAR  = 1 - torch.var(y_true - T, unbiased=False) / (var_y + 1e-8)
        ss_res = torch.sum((T - y_true) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2    = (1 - ss_res / (ss_tot + 1e-8)).item()


        return mse, T
