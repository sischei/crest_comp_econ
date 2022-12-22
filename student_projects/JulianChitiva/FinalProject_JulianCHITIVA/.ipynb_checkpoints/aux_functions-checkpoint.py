# ======================================================================
#
#     sets the auxiliary functions for the "Final Project", i.e., 
#     I/O file functions and equivalences of exotic derivatives 
#     in term of plain vanilla options.
#
#     Julian Chitiva
# ====================================================================== 

import scipy.stats as st
import scipy as sp
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)

from BlackScholes import bsformula
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF

## Define a Portfolio class to handle multiple financial instruments
class Portfolio:
    def __init__(self, path, 
                 lb,
                 ub,
                 training_number = 4,
                 testing_number = 200,
                 n_restarts_optimizer=50,
                 length_scale=1.0,
                 length_scale_bounds=(1e-05, 10000.0),
                 random_state=0):
        self.path = path
        self.df = pd.read_csv(path)
        
        self.lb=lb
        self.ub = ub 
        self.training_number=training_number
        self.testing_number=testing_number
        self.n_restarts_optimizer=n_restarts_optimizer
        self.length_scale=length_scale
        self.length_scale_bounds=length_scale_bounds
        self.random_state=random_state

    def __str__(self):
        df = self.df
        summary = f""" Your portfolio is made up by {df.shape[0]} derivatives with the following characteristics:\n"""
        for i, row in df.iterrows():
            summary += f"""   * {'Long' if row.pos==1 else 'Short'} a {row.type.title()} with Strike(s) {','.join(row.loc[['strike_1','strike_2','strike_3']].dropna().astype(str))} """
            summary += f"""with Underlying Spot {row.S}, time to maturity {row.loc['T']}, risk-free interest rate of {row.r} and implied volatility {row.sigma}.\n"""
        return(summary)
    
    def evaluate_portafolio(self):
        self.portfolio = dict()
        for i, row in self.df.iterrows():
            constructor_fn = eval(f'GP_{row.type.title()}')
            param_dict = {'lb':self.lb, 
                          'ub':self.ub,
                          'sigma':row.sigma,
                          'strike':row.strike_1, 
                          'S0' : row.S,
                          'strike_1':row.strike_1,
                          'strike_2':row.strike_2,
                          'strike_3':row.strike_3,
                          'r':row.r,
                          'T':row.loc['T'],
                          'training_number':self.training_number,
                          'testing_number':self.testing_number,
                          'n_restarts_optimizer':self.n_restarts_optimizer,
                          'length_scale':self.length_scale,
                          'length_scale_bounds':self.length_scale_bounds,
                          'random_state':self.random_state}
            
            instrument=constructor_fn(**param_dict)
            instrument.gaussian_process()
            instrument.test_y = instrument.test_y*(row.pos)
            instrument.train_y = instrument.train_y*(row.pos)
            
            instrument.y_pred = instrument.y_pred*(row.pos)
            instrument.calculate_delta()
            instrument.delta_bs = instrument.delta_bs*(row.pos)
            instrument.f_prime = instrument.f_prime*(row.pos)    
            instrument.calculate_vega()
            instrument.vega_bs = instrument.vega_bs*(row.pos)
            instrument.f_prime2 = instrument.f_prime2*(row.pos)            
            self.portfolio[i]=instrument
            
    def compute_aggregate(self):
        self.test_x = np.array(np.linspace(0, 1, self.testing_number), dtype='float32').reshape(self.testing_number, 1)
        self.test_y = sum(map(lambda x: np.array(x.test_y),self.portfolio.values()))

        self.train_x = np.array(np.linspace(0, 1, self.training_number), dtype='float32').reshape(self.training_number, 1)
        self.train_y = sum(map(lambda x: np.array(x.train_y),self.portfolio.values()))
        
        self.y_pred = sum([x.y_pred for x in self.portfolio.values()])
        self.sigma_pred = np.sqrt(sum([x.sigma_pred**2 for x in self.portfolio.values()]))
        
        self.delta_bs = sum([x.delta_bs for x in self.portfolio.values()])
        self.f_prime = sum([x.f_prime for x in self.portfolio.values()])
        
        self.vega_bs = sum([x.vega_bs for x in self.portfolio.values()])
        self.f_prime2 = sum([x.f_prime2 for x in self.portfolio.values()])
        
    def plot(self):

        fig, ax1 = plt.subplots(figsize = (10,6),
                                facecolor='white', 
                                edgecolor='black')
        lns1=ax1.plot(self.lb+(self.ub-self.lb)*self.test_x.flatten(), 
                 self.test_y, 
                 color = 'black', 
                 label = 'Analytical BS Model',
                 linewidth=1.5)
        lns2=ax1.plot(self.lb+(self.ub-self.lb)*self.test_x.flatten(),
                 self.y_pred, 
                 color = 'red', 
                 label = 'GP Prediction',
                 linewidth = 1)
        ax1.scatter(self.lb+(self.ub-self.lb)*self.train_x, 
                    self.train_y, 
                    color = 'black', 
                    marker = '+', 
                    s = 50) 
        ax1.fill_between(self.lb+(self.ub-self.lb)*self.test_x.flatten(),
                         (self.y_pred.T-2*self.sigma_pred).flatten(), 
                         (self.y_pred.T+2*self.sigma_pred).flatten(), 
                         color = 'grey',
                         alpha=0.3)
        
        ax1.set_xlim([self.lb-10, 
                  self.ub+10])
        ax1.set_xlabel('S')
        ax1.set_ylabel('V')
        
        
        ax2 = ax1.twinx()
        lns3 = ax2.plot(self.lb+(self.ub-self.lb)*self.test_x.flatten(),
                 (self.test_y[:,0]-self.y_pred), 
                        '--',
                 color = 'grey', 
                 label = 'Error')
        ax2.set_ylabel('Error in V')
        
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc = 'best', prop={'size':10})
        plt.show()
        
    def delta_plot(self):
        
        fig, ax1 = plt.subplots(figsize = (10,6),facecolor='white', edgecolor='black')
        
        lns1 = ax1.plot(self.lb+(self.ub-self.lb)*self.test_x, 
                        self.delta_bs,
                        color = 'black', 
                        label = 'Exact Delta')
        lns2 = ax1.plot(self.lb+(self.ub-self.lb)*self.test_x, 
                        self.f_prime, 
                        color = 'red',
                        label = 'GP')
        
        ax2 = ax1.twinx()
        lns3 = ax2.plot(self.lb+(self.ub-self.lb)*self.test_x,
                 self.delta_bs - self.f_prime, 
                        '--',
                 color = 'grey', 
                 label = 'Error')
        
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc = 'best', prop={'size':10})

        
        ax1.set_xlabel('S')
        ax1.set_ylabel('$\delta$')
        ax2.set_ylabel('Error in $\delta$')
        
        plt.show()
        
    def vega_plot(self):
        
        fig, ax1 = plt.subplots(figsize = (10,6),facecolor='white', edgecolor='black')
        
        lns1 = ax1.plot(self.test_x, 
                        self.vega_bs,
                        color = 'black', 
                        label = 'Exact Vega')
        lns2 = ax1.plot(self.test_x, 
                        self.f_prime2, 
                        color = 'red',
                        label = 'GP')
        
        ax2 = ax1.twinx()
        lns3 = ax2.plot(self.test_x,
                 self.vega_bs - self.f_prime2, 
                        '--',
                 color = 'grey', 
                 label = 'Error')
        
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc = 'best', prop={'size':10})

        
        ax1.grid(True)
        ax1.set_xlabel('$\sigma$')
        ax1.set_ylabel('$\\nu$')
        ax2.set_ylabel('Error in $\\nu$')
        
        plt.show()

# Define a Derivative class to define common function to 
# the other financial instruments
class Derivative:
    def __init__(self,
                 lb, # lower bound on domain
                 ub, # upper bound on domain
                 sigma, # implied volatility
                 S0,
                 r, # risk-free rate
                 T, # Time to maturity
                 training_number=5, # Number of training samples
                 testing_number=100, # Number of testing samples
                 n_restarts_optimizer=50,
                 length_scale=1.0,
                 length_scale_bounds=(1e-05, 10000.0),
                 random_state=0,
                 sigma_n = 1e-8 # additive noise in GP
                ):
        self.lb=lb
        self.ub = ub 
        self.sigma = sigma
        self.S0 = S0
        self.r = r
        self.T = T
        self.training_number=training_number
        self.testing_number=testing_number
        self.n_restarts_optimizer=n_restarts_optimizer
        self.length_scale=length_scale
        self.length_scale_bounds=length_scale_bounds
        self.random_state=random_state
        self.sigma_n = sigma_n
    
    def training(self):
        self.train_x = np.array(np.linspace(0.01, 1, self.training_number), dtype='float32').reshape(self.training_number, 1)
        self.train_y = np.array([self.function(S, self.sigma) for S in self.train_x])
    
    def testing(self):
        self.test_x = np.array(np.linspace(0.01, 1, self.testing_number), dtype='float32').reshape(self.testing_number, 1)
        self.test_y =  np.array([self.function(S, self.sigma) for S in self.test_x])
    
    def gaussian_process(self):
        ## Execute the functions to have the values needed
        self.training()
        self.testing()
        
        ## Execute the gaussian process regression
        sk_kernel = RBF(length_scale=self.length_scale, 
                        length_scale_bounds=self.length_scale_bounds)
        self.gp = gaussian_process.GaussianProcessRegressor(kernel=sk_kernel,
                                                            n_restarts_optimizer=self.n_restarts_optimizer,
                                                            random_state = self.random_state)
        self.gp.fit(self.train_x,self.train_y)
        self.y_pred, self.sigma_pred = self.gp.predict(self.test_x, return_std=True)
        
    def plot(self):
        fig, ax1 = plt.subplots(figsize = (10,6),
                                facecolor='white', 
                                edgecolor='black')
        lns1=ax1.plot(self.lb+(self.ub-self.lb)*self.test_x.flatten(), 
                 self.test_y, 
                 color = 'black', 
                 label = 'Analytical Model',
                 linewidth=1.5)
        lns2=ax1.plot(self.lb+(self.ub-self.lb)*self.test_x.flatten(),
                 self.y_pred, 
                 color = 'red', 
                 label = 'GP Prediction',
                 linewidth = 1)
        ax1.scatter(self.lb+(self.ub-self.lb)*self.train_x, 
                    self.train_y, 
                    color = 'black', 
                    marker = '+', 
                    s = 50) 
        ax1.fill_between(self.lb+(self.ub-self.lb)*self.test_x.flatten(),
                         (self.y_pred.T-2*self.sigma_pred).flatten(), 
                         (self.y_pred.T+2*self.sigma_pred).flatten(), 
                         color = 'grey',
                         alpha=0.3)
        
        ax1.set_xlim([self.lb-10, 
                  self.ub+10])
        ax1.set_xlabel('S')
        ax1.set_ylabel('V')
        
        
        ax2 = ax1.twinx()
        lns3 = ax2.plot(self.lb+(self.ub-self.lb)*self.test_x.flatten(),
                 (self.test_y[:,0]-self.y_pred), 
                        '--',
                 color = 'grey', 
                 label = 'GP Error')
        ax2.set_ylabel('Error in V')
        
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc = 'best', prop={'size':10})
        plt.show()
      
    def delta_vega(self):
        l = self.gp.kernel_.length_scale
        rbf = RBF(length_scale=l)
        Kernel = rbf(self.train_x, self.train_x)
        
        K_y= Kernel + np.eye(self.training_number)*self.sigma_n
        L = sp.linalg.cho_factor(K_y)
        alpha_p = sp.linalg.cho_solve(np.transpose(L), self.train_y)
        
        k_s = rbf(self.test_x, self.train_x)
        
        # Delta
        k_s_prime = (self.train_x.T - self.test_x) * k_s / l**2
        self.f_prime = np.dot(k_s_prime, alpha_p) / (self.ub - self.lb)
        
        # Vega 
        
        self.train_y2 = []
        for idx in range(len(self.train_x)):
            self.train_y2.append(self.function((self.S0-self.lb)/(self.ub-self.lb), self.train_x[idx]))
        self.train_y2 = np.array(self.train_y2)
        
        sk_kernel2 = RBF(length_scale=self.length_scale,
                         length_scale_bounds=self.length_scale_bounds)
        self.gp2 = gaussian_process.GaussianProcessRegressor(kernel=sk_kernel2, 
                                                             n_restarts_optimizer=self.n_restarts_optimizer)
    
        self.gp2.fit(self.train_x, self.train_y2)
        self.y_pred2, self.sigma_hat2 = self.gp2.predict(self.test_x, return_std=True)
        l = self.gp2.kernel_.length_scale
        rbf = gaussian_process.kernels.RBF(length_scale=l)

        Kernel= rbf(self.train_x, self.train_x)
        K_y = Kernel + np.eye(self.training_number) * self.sigma_n
        L = sp.linalg.cho_factor(K_y)
        alpha_p = sp.linalg.cho_solve(np.transpose(L), self.train_y2)

        k_s = rbf(self.test_x, self.train_x)

        k_s_prime_2 = np.zeros([len(self.test_x), len(self.train_x)])
        for i in range(len(self.test_x)):
            for j in range(len(self.train_x)):
                k_s_prime_2[i, j] = (1.0/l**2) * (self.train_x[j] - self.test_x[i]) * k_s[i, j]

        self.f_prime2 = np.dot(k_s_prime_2, alpha_p)
        
    def delta_plot(self):
        
        fig, ax1 = plt.subplots(figsize = (10,6),facecolor='white', edgecolor='black')
        
        lns1 = ax1.plot(self.lb+(self.ub-self.lb)*self.test_x, 
                        self.delta_bs,
                        color = 'black', 
                        label = 'Exact')
        lns2 = ax1.plot(self.lb+(self.ub-self.lb)*self.test_x, 
                        self.f_prime, 
                        color = 'red',
                        label = 'GP')
        
        ax2 = ax1.twinx()
        lns3 = ax2.plot(self.lb+(self.ub-self.lb)*self.test_x,
                 self.delta_bs - self.f_prime, 
                        '--',
                 color = 'grey', 
                 label = 'GP Error')
        
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc = 'best', prop={'size':10})

        
        ax1.grid(True)
        ax1.set_xlabel('S')
        ax1.set_ylabel('$\Delta$')
        ax2.set_ylabel('Error in $\Delta$')
        
        plt.show()
        
    def vega_plot(self):
        
        fig, ax1 = plt.subplots(figsize = (10,6),facecolor='white', edgecolor='black')
        
        lns1 = ax1.plot(self.test_x, 
                        self.vega_bs,
                        color = 'black', 
                        label = 'Exact')
        lns2 = ax1.plot(self.test_x, 
                        self.f_prime2, 
                        color = 'red',
                        label = 'GP')
        
        ax2 = ax1.twinx()
        lns3 = ax2.plot(self.test_x,
                 self.vega_bs - self.f_prime2, 
                        '--',
                 color = 'grey', 
                 label = 'GP Error')
        
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc = 'best', prop={'size':10})

        
        ax1.grid(True)
        ax1.set_xlabel('$\sigma$')
        ax1.set_ylabel('$\\nu$')
        ax2.set_ylabel('Error in $\\nu$')
        
        plt.show()


# Define a Call class that extends Derivative to specific functionalities 
class GP_Call(Derivative):
    def __init__(self, 
                 lb,
                 ub,
                 sigma,
                 S0,
                 strike,
                 r,
                 T,
                 training_number=5, 
                 testing_number=100,
                 n_restarts_optimizer=20,
                 length_scale=1.0,
                 length_scale_bounds=(1e-05, 10000.0),
                 random_state=0,
                 sigma_n = 1e-8, # additive noise in GP
                 **kwargs):
        
        super().__init__(lb=lb, 
                         ub=ub, 
                         sigma=sigma, 
                         S0=S0,
                         r=r, 
                         T=T, 
                         training_number=training_number, 
                         testing_number=testing_number, 
                         n_restarts_optimizer=n_restarts_optimizer,
                         length_scale=length_scale,
                         length_scale_bounds=length_scale_bounds,
                         random_state=random_state,
                         sigma_n=sigma_n)
        
        self.strike = strike
        
        self.function = lambda x,y: bsformula(1, 
                                            self.lb+(self.ub-self.lb)*x, 
                                            self.strike, 
                                            self.r,
                                            self.T,
                                              y, 0)[0]
        
        self.function_delta = lambda x: bsformula(1,
                                                  self.lb+(self.ub-self.lb)*x, 
                                                  self.strike,
                                                  self.r, 
                                                  self.T, 
                                                  self.sigma, 0)[1]
        self.function_vega = lambda x,y: bsformula(1,
                                                  self.lb+(self.ub-self.lb)*x, 
                                                  self.strike,
                                                  self.r, 
                                                  self.T, 
                                                  y, 0)[2]
        
        
    def calculate_delta(self):
        self.delta_vega()        
        self.delta_bs=self.function_delta(self.test_x)
        
    def delta_plot(self):
        self.calculate_delta()
        super().delta_plot()
        
    def calculate_vega(self):
        self.delta_vega()        
        self.vega_bs=self.function_vega((self.S0-self.lb)/(self.ub-self.lb),self.test_x)
      
    def vega_plot(self):
        self.calculate_vega()
        super().vega_plot()


# Define a Put class that extends Derivative to specific functionalities        
class GP_Put(Derivative):
    def __init__(self, 
                 lb,
                 ub,
                 sigma,
                 S0,
                 strike,
                 r,
                 T,
                 training_number=5, 
                 testing_number=100,
                 n_restarts_optimizer=20,
                 length_scale=1.0,
                 length_scale_bounds=(1e-05, 10000.0),
                 random_state=0,
                 sigma_n = 1e-8, # additive noise in GP
                 **kwargs):
        
        super().__init__(lb=lb, 
                         ub=ub, 
                         sigma=sigma,
                         S0=S0,
                         r=r, 
                         T=T, 
                         training_number=training_number, 
                         testing_number=testing_number, 
                         n_restarts_optimizer=n_restarts_optimizer,
                         length_scale=length_scale,
                         length_scale_bounds=length_scale_bounds,
                         random_state=random_state,
                         sigma_n=sigma_n)
        
        
        self.strike = strike
        
        self.function = lambda x,y: bsformula(-1, 
                                            self.lb+(self.ub-self.lb)*x, 
                                            self.strike,
                                            self.r, 
                                            self.T, 
                                            y, 0)[0]
        self.function_delta = lambda x: bsformula(-1,
                                                  self.lb+(self.ub-self.lb)*x, 
                                                  self.strike,
                                                  self.r, 
                                                  self.T, 
                                                  self.sigma, 0)[1]
        
        self.function_vega = lambda x,y: bsformula(-1,
                                                  self.lb+(self.ub-self.lb)*x, 
                                                  self.strike,
                                                  self.r, 
                                                  self.T, 
                                                  y, 0)[2]
        
    def calculate_delta(self):
        self.delta_vega()
        self.delta_bs=self.function_delta(self.test_x)
        
    def delta_plot(self):
        self.calculate_delta()
        super().delta_plot()

    def calculate_vega(self):
        self.delta_vega()        
        self.vega_bs=self.function_vega((self.S0-self.lb)/(self.ub-self.lb),self.test_x)
      
    def vega_plot(self):
        self.calculate_vega()
        super().vega_plot()

# Define a Bull class that extends Derivative to specific functionalities       
class GP_Bull(Derivative):
    def __init__(self, 
                 lb,
                 ub,
                 sigma,
                 S0,
                 strike_1,
                 strike_2,                 
                 r,
                 T,
                 training_number=5, 
                 testing_number=100,
                 n_restarts_optimizer=20,
                 length_scale=1.0,
                 length_scale_bounds=(1e-05, 10000.0),
                 random_state=0,
                 sigma_n = 1e-8, # additive noise in GP
                 **kwargs):
        super().__init__(lb=lb, 
                         ub=ub, 
                         sigma=sigma, 
                         S0=S0,
                         r=r, 
                         T=T, 
                         training_number=training_number, 
                         testing_number=testing_number, 
                         n_restarts_optimizer=n_restarts_optimizer,
                         length_scale=length_scale,
                         length_scale_bounds=length_scale_bounds,
                         random_state=random_state,
                         sigma_n=sigma_n)
        
        self.strike_1 = strike_1
        self.strike_2 = strike_2
        
        self.long_call = GP_Call(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                                 S0=self.S0,
                                 strike=self.strike_1, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)
        self.short_call = GP_Call(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                                  S0=self.S0,
                                 strike=self.strike_2, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)

    def testing(self):
        self.test_x = np.array(np.linspace(0, 1, self.testing_number), 
                               dtype='float32').reshape(self.testing_number, 1)
        self.test_y = np.array(self.long_call.test_y)- np.array(self.short_call.test_y)

    def training(self):
        self.train_x = np.array(np.linspace(0, 1, self.training_number), 
                                dtype='float32').reshape(self.training_number, 1)
        self.train_y = np.array(self.long_call.train_y)- np.array(self.short_call.train_y)
        
    def gaussian_process(self):
        self.long_call.gaussian_process()
        self.short_call.gaussian_process()
        self.testing()
        self.training()
        
        # The Gaussian process' predictions
        self.y_pred = self.long_call.y_pred - self.short_call.y_pred
        self.sigma_pred = np.sqrt(self.long_call.sigma_pred**2 + self.short_call.sigma_pred**2)
        
     
    def calculate_delta(self):
        self.long_call.calculate_delta()
        self.short_call.calculate_delta()
        self.delta_bs=self.long_call.delta_bs-self.short_call.delta_bs
        self.f_prime=self.long_call.f_prime-self.short_call.f_prime
        
    def delta_plot(self):
        self.calculate_delta()
        super().delta_plot()
        
    def calculate_vega(self):
        self.long_call.calculate_vega()
        self.short_call.calculate_vega()
        self.vega_bs=self.long_call.vega_bs-self.short_call.vega_bs
        self.f_prime2=self.long_call.f_prime2-self.short_call.f_prime2
      
    def vega_plot(self):
        self.calculate_vega()
        super().vega_plot()


# Define a Bear class that extends Derivative to specific functionalities       
class GP_Bear(Derivative):
    def __init__(self, 
                 lb,
                 ub,
                 sigma,
                 S0,
                 strike_1,
                 strike_2,                 
                 r,
                 T,
                 training_number=5, 
                 testing_number=100,
                 n_restarts_optimizer=20,
                 length_scale=1.0,
                 length_scale_bounds=(1e-05, 10000.0),
                 random_state=0,
                 sigma_n = 1e-8, # additive noise in GP
                 **kwargs):
        super().__init__(lb=lb, 
                         ub=ub, 
                         sigma=sigma, 
                         S0=S0,
                         r=r, 
                         T=T, 
                         training_number=training_number, 
                         testing_number=testing_number, 
                         n_restarts_optimizer=n_restarts_optimizer,
                         length_scale=length_scale,
                         length_scale_bounds=length_scale_bounds,
                         random_state=random_state,
                         sigma_n=sigma_n)
        
        self.strike_1 = strike_1
        self.strike_2 = strike_2
        
        self.long_put = GP_Put(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                               S0=self.S0,
                                 strike=self.strike_2, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)
        self.short_put = GP_Put(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                                S0=self.S0,
                                 strike=self.strike_1, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)

    def testing(self):
        self.test_x = np.array(np.linspace(0, 1, self.testing_number), 
                               dtype='float32').reshape(self.testing_number, 1)
        self.test_y = np.array(self.long_put.test_y)- np.array(self.short_put.test_y)

    def training(self):
        self.train_x = np.array(np.linspace(0, 1, self.training_number), 
                                dtype='float32').reshape(self.training_number, 1)
        self.train_y = np.array(self.long_put.train_y)- np.array(self.short_put.train_y)
        
    def gaussian_process(self):
        self.long_put.gaussian_process()
        self.short_put.gaussian_process()
        self.testing()
        self.training()
        
        # The Gaussian process' predictions
        self.y_pred = self.long_put.y_pred - self.short_put.y_pred
        self.sigma_pred = np.sqrt(self.long_put.sigma_pred**2 + self.short_put.sigma_pred**2)
        
    def calculate_delta(self):
        self.long_put.calculate_delta()
        self.short_put.calculate_delta()
        self.delta_bs=self.long_put.delta_bs-self.short_put.delta_bs
        self.f_prime=self.long_put.f_prime-self.short_put.f_prime
        
    def delta_plot(self):
        self.calculate_delta()
        super().delta_plot()
        
    def calculate_vega(self):
        self.long_put.calculate_vega()
        self.short_put.calculate_vega()
        self.vega_bs=self.long_put.vega_bs-self.short_put.vega_bs
        self.f_prime2=self.long_put.f_prime2-self.short_put.f_prime2
      
    def vega_plot(self):
        self.calculate_vega()
        super().vega_plot()

# Define a Straddle class that extends Derivative to specific functionalities
class GP_Straddle(Derivative):
    def __init__(self, 
                 lb,
                 ub,
                 sigma,
                 S0,
                 strike,
                 r,
                 T,
                 training_number=5, 
                 testing_number=100,
                 n_restarts_optimizer=20,
                 length_scale=1.0,
                 length_scale_bounds=(1e-05, 10000.0),
                 random_state=0,
                 sigma_n = 1e-8,# additive noise in GP
                 **kwargs):
        super().__init__(lb=lb, 
                         ub=ub, 
                         sigma=sigma,
                         S0=S0,
                         r=r, 
                         T=T, 
                         training_number=training_number, 
                         testing_number=testing_number, 
                         n_restarts_optimizer=n_restarts_optimizer,
                         length_scale=length_scale,
                         length_scale_bounds=length_scale_bounds,
                         random_state=random_state,
                         sigma_n=sigma_n)

        self.strike = strike
        
        self.long_put = GP_Put(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                               S0=self.S0,
                                 strike=self.strike, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)
        self.long_call = GP_Call(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                                 S0=self.S0,
                                 strike=self.strike, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)

    def testing(self):
        self.test_x = np.array(np.linspace(0, 1, self.testing_number), 
                               dtype='float32').reshape(self.testing_number, 1)
        self.test_y = np.array(self.long_put.test_y)+np.array(self.long_call.test_y)

    def training(self):
        self.train_x = np.array(np.linspace(0, 1, self.training_number), 
                                dtype='float32').reshape(self.training_number, 1)
        self.train_y = np.array(self.long_put.train_y)+np.array(self.long_call.train_y)
        
    def gaussian_process(self):
        self.long_put.gaussian_process()
        self.long_call.gaussian_process()
        self.testing()
        self.training()
        
        # The Gaussian process' predictions
        self.y_pred = self.long_put.y_pred + self.long_call.y_pred
        self.sigma_pred = np.sqrt(self.long_put.sigma_pred**2 + self.long_call.sigma_pred**2)
        
    
    def calculate_delta(self):
        self.long_put.calculate_delta()
        self.long_call.calculate_delta()
        self.delta_bs=self.long_put.delta_bs+self.long_call.delta_bs
        self.f_prime=self.long_put.f_prime+self.long_call.f_prime
        
    def delta_plot(self):
        self.calculate_delta()
        super().delta_plot()
        
    def calculate_vega(self):
        self.long_put.calculate_vega()
        self.long_call.calculate_vega()
        self.vega_bs=self.long_put.vega_bs+self.long_call.vega_bs
        self.f_prime2=self.long_put.f_prime2+self.long_call.f_prime2
      
    def vega_plot(self):
        self.calculate_vega()
        super().vega_plot()


# Define a Strangle class that extends Derivative to specific functionalities
class GP_Strangle(Derivative):
    def __init__(self, 
                 lb,
                 ub,
                 sigma,
                 S0,
                 strike_1,
                 strike_2,                
                 r,
                 T,
                 training_number=5, 
                 testing_number=100,
                 n_restarts_optimizer=20,
                 length_scale=1.0,
                 length_scale_bounds=(1e-05, 10000.0),
                 random_state=0,
                 sigma_n = 1e-8, # additive noise in GP
                 **kwargs):
        super().__init__(lb=lb, 
                         ub=ub, 
                         sigma=sigma, 
                         S0=S0,
                         r=r, 
                         T=T, 
                         training_number=training_number, 
                         testing_number=testing_number, 
                         n_restarts_optimizer=n_restarts_optimizer,
                         length_scale=length_scale,
                         length_scale_bounds=length_scale_bounds,
                         random_state=random_state,
                         sigma_n=sigma_n)

        self.strike_1 = strike_1
        self.strike_2 = strike_2
        
        self.long_put = GP_Put(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                               S0=self.S0,
                                 strike=self.strike_1, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)
        self.long_call = GP_Call(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                                 S0=self.S0,
                                 strike=self.strike_2, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)

    def testing(self):
        self.test_x = np.array(np.linspace(0, 1, self.testing_number), dtype='float32').reshape(self.testing_number, 1)
        self.test_y = np.array(self.long_put.test_y)+np.array(self.long_call.test_y)

    def training(self):
        self.train_x = np.array(np.linspace(0, 1, self.training_number), dtype='float32').reshape(self.training_number, 1)
        self.train_y = np.array(self.long_put.train_y)+np.array(self.long_call.train_y)
        
    def gaussian_process(self):
        self.long_put.gaussian_process()
        self.long_call.gaussian_process()
        self.testing()
        self.training()
        
        # The Gaussian process' predictions
        self.y_pred = self.long_put.y_pred + self.long_call.y_pred
        self.sigma_pred = np.sqrt(self.long_put.sigma_pred**2 + self.long_call.sigma_pred**2)
        
    def calculate_delta(self):
        self.long_put.calculate_delta()
        self.long_call.calculate_delta()
        self.delta_bs=self.long_put.delta_bs+self.long_call.delta_bs
        self.f_prime=self.long_put.f_prime+self.long_call.f_prime
        
    def delta_plot(self):
        self.calculate_delta()
        super().delta_plot()
        
    def calculate_vega(self):
        self.long_put.calculate_vega()
        self.long_call.calculate_vega()
        self.vega_bs=self.long_put.vega_bs+self.long_call.vega_bs
        self.f_prime2=self.long_put.f_prime2+self.long_call.f_prime2
        
    def vega_plot(self):
        self.calculate_vega()
        super().vega_plot()


# Define a Butterfly class that extends Derivative to specific functionalities  
class GP_Butterfly(Derivative):
    def __init__(self, 
                 lb,
                 ub,
                 sigma,
                 S0,
                 strike_1,
                 strike_2,                
                 strike_3,
                 r,
                 T,
                 training_number=5, 
                 testing_number=100,
                 n_restarts_optimizer=20,
                 length_scale=1.0,
                 length_scale_bounds=(1e-05, 10000.0),
                 random_state=0,
                 sigma_n = 1e-8, # additive noise in GP
                 **kwargs):
        super().__init__(lb=lb, 
                         ub=ub, 
                         sigma=sigma,
                         S0=S0,
                         r=r, 
                         T=T, 
                         training_number=training_number, 
                         testing_number=testing_number, 
                         n_restarts_optimizer=n_restarts_optimizer,
                         length_scale=length_scale,
                         length_scale_bounds=length_scale_bounds,
                         random_state=random_state,
                         sigma_n=sigma_n)

        self.strike_1 = strike_1
        self.strike_2 = strike_2
        self.strike_3 = strike_3
        
        self.long_call = GP_Put(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                                S0=self.S0,
                                 strike=self.strike_1, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)
        self.short_call = GP_Call(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                                  S0=self.S0,
                                 strike=self.strike_2, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)
        self.long_call2 = GP_Call(lb= self.lb, 
                                 ub= self.ub,
                                 sigma=self.sigma,
                                  S0=self.S0,
                                 strike=self.strike_3, 
                                 r=self.r,
                                 T= self.T,
                                 training_number=self.training_number,
                                 testing_number=self.testing_number,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 length_scale=self.length_scale,
                                 length_scale_bounds=self.length_scale_bounds)

    def testing(self):
        self.test_x = np.array(np.linspace(0, 1, self.testing_number), 
                               dtype='float32').reshape(self.testing_number, 1)
        self.test_y = np.array(self.long_call.test_y)-2*np.array(self.short_call.test_y)+np.array(self.long_call2.test_y)

    def training(self):
        self.train_x = np.array(np.linspace(0, 1, self.training_number), 
                                dtype='float32').reshape(self.training_number, 1)
        self.train_y = np.array(self.long_call.train_y)-2*np.array(self.short_call.train_y)+np.array(self.long_call2.train_y)
        
    def gaussian_process(self):
        self.long_call.gaussian_process()
        self.short_call.gaussian_process()
        self.long_call2.gaussian_process()
        self.testing()
        self.training()
        
        # The Gaussian process' predictions
        self.y_pred = self.long_call.y_pred -2*self.short_call.y_pred + self.long_call2.y_pred
        self.sigma_pred = np.sqrt(self.long_call.sigma_pred**2 + self.short_call.sigma_pred**2 + self.long_call2.sigma_pred**2)        
        
        
    def calculate_delta(self):
        self.long_call.calculate_delta()
        self.short_call.calculate_delta()
        self.long_call2.calculate_delta()
        self.delta_bs=self.long_call.delta_bs-2*self.short_call.delta_bs+self.long_call2.delta_bs
        self.f_prime=self.long_call.f_prime-2*self.short_call.f_prime+self.long_call2.f_prime
        
    def delta_plot(self):
        self.calculate_delta()
        super().delta_plot()
        
    def calculate_vega(self):
        self.long_call.calculate_vega()
        self.short_call.calculate_vega()
        self.long_call2.calculate_vega()
        self.vega_bs=self.long_call.vega_bs-2*self.short_call.vega_bs+self.long_call2.vega_bs
        self.f_prime2=self.long_call.f_prime2-2*self.short_call.f_prime2+self.long_call2.f_prime2
        
    def vega_plot(self):
        self.calculate_vega()
        super().vega_plot()
