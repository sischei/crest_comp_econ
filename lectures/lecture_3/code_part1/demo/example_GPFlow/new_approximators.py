import gpflow
import tensorflow as tf
tftype=tf.float64
gpflow.config.set_default_float(tftype)
import numpy as np
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary

            

class GPflow_norm():
    def __init__(self, Xdata, Ydata, kernel_name = 'RBF', max_opt_iter = 100, optimize = True, lengthscales_init = None, variances_init = None, likelihood_variance_init = None, params_dict = None, lvar_trainable = False):
        Xdatashape = Xdata.shape
        Ydatashape = Ydata.shape


        self.Ndata_tot = Xdatashape[0]
        self.Ndata_usedstats = Xdatashape[0]
        
        self.dimX = Xdatashape[1]

        assert Ydatashape[0] == self.Ndata_tot, 'need same amount of Xdata and Ydata'
        assert Ydatashape[1] == 1, 'shape of Ydata should be (N, 1)'

        self.Xdata = tf.constant(Xdata, dtype = tftype)
        self.Ydata = tf.constant(Ydata, dtype = tftype)

        self.Xmean = tf.math.reduce_mean(self.Xdata, axis = 0, keepdims = True)
        self.Xstd = tf.math.reduce_std(self.Xdata, axis = 0, keepdims = True)
        self.Xstd = tf.math.maximum(self.Xstd, 0.001 * tf.ones_like(self.Xstd), name=None)

        self.Ymean = tf.math.reduce_mean(self.Ydata, axis = 0, keepdims = True)
        self.Ystd = tf.math.reduce_std(self.Ydata, axis = 0, keepdims = True)
        self.Ystd = tf.math.maximum(self.Ystd, 0.001 * tf.ones_like(self.Ystd), name=None)

        self.norm_Xdata = self.normalize_X(self.Xdata)
        self.norm_Ydata = self.normalize_Y(self.Ydata)

        self.lvar_trainable = lvar_trainable

        self.kernel_name = kernel_name

        if np.any(lengthscales_init == None):
            lengthscales_init = tf.ones(self.dimX, dtype = tftype)
        if np.any(variances_init == None):
            variances_init = tf.constant(1., dtype=tftype)

        if self.kernel_name == 'Matern52':
            self.kernel = gpflow.kernels.Matern52(lengthscales=lengthscales_init, variance=variances_init)
        elif self.kernel_name == 'RBF':
            self.kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales_init, variance=variances_init) 
        else:
            assert False, 'only Matern52 and RBF implemented at the moment.'

        self.max_opt_iter = max_opt_iter

        self.opt = gpflow.optimizers.Scipy()
        
        self.model = gpflow.models.GPR(data=(self.norm_Xdata, self.norm_Ydata), kernel=self.kernel, mean_function=None)
        self.model.likelihood.variance.assign(tf.constant(4.0e-6, dtype=tftype))
        #print(self.model.trainable_variables)
        gpflow.set_trainable(self.model.likelihood.variance, self.lvar_trainable)


        if not(params_dict == None):
            #print('here')
            gpflow.utilities.multiple_assign(self.model, params_dict)

        if optimize:
            self.optimize_hp(self.max_opt_iter)
                                                
        
    def optimize_hp(self, max_opt_iter):
        self.max_opt_iter = max_opt_iter
        self.opt.minimize(self.model.training_loss, self.model.trainable_variables, options=dict(maxiter=self.max_opt_iter))

    def print_summary(self):
        print_summary(self.model)

    def normalize_X(self, Xdata):
        return (Xdata - self.Xmean) / self.Xstd
    
    def normalize_Y(self, Ydata):
        return (Ydata - self.Ymean) / self.Ystd

    def denormalize_X(self, Xdata):
        return Xdata * self.Xstd + self.Xmean
    
    def denormalize_Y(self, Ydata):
        return Ydata * self.Ystd + self.Ymean

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, Xdata):
        assert Xdata.shape[1] == self.dimX, 'input data should be of shape (?, {})'.format(self.dimX)
        normXdata = self.normalize_X(Xdata)

        pred_mean_norm, _ = self.model.predict_f(normXdata)
        pred_mean = self.denormalize_Y(pred_mean_norm)

        return pred_mean

    def predict(self, Xdata):
        return self._predict(tf.constant(Xdata, dtype=tftype))

    @tf.function(experimental_relax_shapes=True)
    def _predict_with_var(self, Xdata):
        assert Xdata.shape[1] == self.dimX, 'input data should be of shape (?, {})'.format(self.dimX)
        
        normXdata = self.normalize_X(Xdata)

        pred_mean_norm, pred_var_norm = self.model.predict_f(normXdata)
        pred_mean = self.denormalize_Y(pred_mean_norm)
        
        pred_var = pred_var_norm * self.Ystd ** 2.

        return pred_mean, pred_var

    def predict_with_var(self, Xdata):
        return self._predict_with_var(tf.constant(Xdata, dtype=tftype))

    def get_model_params(self):
        return gpflow.utilities.read_values(self.model)

        





