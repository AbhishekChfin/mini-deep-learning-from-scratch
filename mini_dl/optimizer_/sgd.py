import numpy as np

from mini_dl.core import parameter

"""
SGD
 ├── Momentum
 ├── AdaGrad
      └── RMSProp
             └── Adam (Momentum + RMSProp)
"""

class SGD:

    def __init__(self, learning_rate=1., decay=0., momentum=0., adagrad=False):
        self.current_lr = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.adagrad = adagrad

    def _parameters_from(self, layers):
        layers = layers if isinstance(layers, (list, tuple)) else [layers]
        parameters = []
        for layer in layers:
            layer.parameters()
            parameters.extend(layer.trainable_parameters)
        return parameters
    
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1./(1. + (self.decay * self.iterations)))
        
    def step(self, layers):
        self.pre_update_params()

        for parameter in self._parameters_from(layers):
            if self.momentum:
                # if self.adagrad:
                #     if not hasattr(parameter, 'cache'):
                #         parameter.cache = np.zeros_like(parameter.data)
                #     if not hasattr(parameter, 'momentum'):
                #         parameter.momentum = np.zeros_like(parameter.data)
                #     parameter.cache += parameter.grad**2
                #     parameter_update = self.momentum * parameter.momentum - (self.current_lr * parameter.grad) / \
                #     (np.sqrt(parameter.cache) + 1e-7)
                #     parameter.momentum = parameter_update
                #     parameter.data += parameter_update
                    
                # else:
                if not hasattr(parameter, 'momentum'):
                    parameter.momentum = np.zeros_like(parameter.data)

                parameter_update = self.momentum * parameter.momentum - self.current_lr * parameter.grad
                parameter.momentum = parameter_update
                parameter.data += parameter_update
            
            
            elif self.adagrad:
                if not hasattr(parameter, 'cache'):
                    parameter.cache = np.zeros_like(parameter.data)
                parameter.cache += parameter.grad**2
                parameter_update = -(self.current_lr * parameter.grad) / \
                    (np.sqrt(parameter.cache) + 1e-7)
                parameter.data += parameter_update
                    
            else:
                parameter.data -= self.current_lr * parameter.grad

        self.post_update_params()
        
        
    
    def post_update_params(self):
        self.iterations += 1

    def zero_grad(self, layers):
        for parameter in self._parameters_from(layers):
            parameter.zero_grad()


class Optimizer_adagrad:
    
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_lr = learning_rate
        # self.adagrad = self.adagrad
        self.iteration = 0

    def _parameters_from(self, layers):
        layers = layers if isinstance(layers, (list, tuple)) else [layers]
        parameters = []
        for layer in layers:
            layer.parameters()
            parameters.extend(layer.trainable_parameters)
        return parameters
    
    def _pre_update_param(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1./(1+self.decay*self.iteration))
    

    def step(self, layers):
        self._pre_update_param()

        for parameter in self._parameters_from(layers):
            if not hasattr(parameter, 'cache'):
                parameter.cache = np.zeros_like(parameter.data)
            parameter.cache += parameter.grad**2
            parameter_update = -(self.current_lr * parameter.grad) / \
                (np.sqrt(parameter.cache + 1e-7))
            parameter.data += parameter_update
            
        
        self._post_update_param()

    def _post_update_param(self):
        self.iteration+=1
    


class RMSProp:
    def __init__(self, learning_rate=1., decay=0., rho=0.):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.rho = rho
        self.decay = decay
        self.iteration = 0
    
    def _parameter_from(self, layers):
        layers = layers if isinstance(layers, (list, tuple)) else [layers]
        parameters = []
        for layer in layers:
            layer.parameters()
            parameters.extend(layer.trainable_parameters)
        return parameters
    
    def _pre_update_param(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1. / (1 + self.decay*self.iteration))
    
    def step(self, layers):
        self._pre_update_param()

        for parameter in self._parameter_from(layers):
            if not hasattr(parameter, 'cache'):
                parameter.cache = np.zeros_like(parameter.grad)
            parameter.cache = self.rho * parameter.cache + (1-self.rho) * (parameter.grad**2)
            update = -self.current_lr*parameter.grad / \
                (np.sqrt(parameter.cache) + 1e-7)
            parameter.data += update
    
        self._post_update_param()
    
    def _post_update_param(self):
        self.iteration += 1
    


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0., beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay

        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.iteration = 1
    
    def _parameter_from(self, layers):
        layers = layers if isinstance(layers, (list, tuple)) else [layers]
        parameters = []
        for layer in layers:
            layer.parameters()
            parameters.extend(layer.trainable_parameters)
        return parameters
    
    def _pre_update_param(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1. / (1 + self.decay*self.iteration))
    
    def step(self, layers):
        self._pre_update_param()

        for parameter in self._parameter_from(layers):
            if not hasattr(parameter, 'cache'):
                parameter.cache = np.zeros_like(parameter.grad)
            if not hasattr(parameter, 'momentum'):
                parameter.momentum = np.zeros_like(parameter.grad)

            # momentum term
            parameter.momentum = self.beta_1*parameter.momentum + \
                (1 - self.beta_1) * parameter.grad
            
            # corrected momentum so that the intial few moments are not bias towards intial momentum value that is zero
            m_hat = parameter.momentum / (1 - self.beta_1**self.iteration)
            
            # cache term
            parameter.cache = self.beta_2 * parameter.cache + \
                (1 - self.beta_2) * (parameter.grad**2)
            
            # corrected cache term so the intital few caches are not bias toward their intial values that is zero
            v_hat = parameter.cache / (1 - self.beta_2**self.iteration)

            # update params
            parameter.data += -self.current_lr * (m_hat) / \
                np.sqrt(v_hat + self.epsilon)
    
        self._post_update_param()
    
    def _post_update_param(self):
        self.iteration += 1
    

    