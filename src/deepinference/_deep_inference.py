from deepinference import __version__

__author__ = "Andrea Ravenni"
__copyright__ = "Andrea Ravenni"
__license__ = "MIT"


import tensorflow as _tf
from tensorflow.keras import Model as _Model
from tensorflow.keras import layers as _layers
from tensorflow.keras import Input as _Input
from tensorflow.keras import regularizers as _regularizers

import keras_tuner as _kt

from . import losses
from . import activations

from .CLR import CyclicLR

# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from deepinference.skeleton import fib`,
# when using this Python module as a library.

class Model(_Model):
    """basic wrapper of the model
    overwrite fit so that the training and validation dataset do not need to be specified
    """   
    def __init__(self,
                 inputs, outputs, 
                 training_set_properties,
                 training_set_labels,
                 validation_set_properties,
                 validation_set_labels,
                 **kwargs
    ):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.training_set_properties = training_set_properties #TODO these should be shallow copies!
        self.training_set_labels = training_set_labels
        self.validation_set_properties = validation_set_properties
        self.validation_set_labels = validation_set_labels
        
    def fit(
            self,
            **kwargs,
    ):
        return super().fit(
            self.training_set_properties,
            self.training_set_labels,
            validation_data=(self.validation_set_properties,
                             self.validation_set_labels),
            **kwargs,
        )

class ModelBuilder:
    """TODO: add description
    """
    
    def __init__(
            self, 
            training_set_properties,
            training_set_labels,
            validation_set_properties,
            validation_set_labels,
    ):
        self.training_set_properties = training_set_properties #TODO these should be shallow copies!
        self.training_set_labels = training_set_labels
        self.validation_set_properties = validation_set_properties
        self.validation_set_labels = validation_set_labels
        
        self.normalize_layer = _layers.Normalization()
        self.normalize_layer.adapt(self.training_set_properties)

    def __build_network_with_given_output_layer(
            self,
            output_layer_func,
            initializer,
            loss,
            nodes_hidden_layers=[256,256],
            reg_L2_weight=0.,
            reg_WD_rate=None,
            dropout_rate=0.,
            do_batch_normalization=False,
            hidden_layer_activation='leaky_relu',
            Adam_learning_rate=1.e-5,
            **kwargs
    ):
        inputs = _Input(shape=self.training_set_properties.shape[1])

        norm_inputs = self.normalize_layer(inputs)
        layer = norm_inputs
        
        for n_nodes in nodes_hidden_layers:
            tlayer = _layers.Dense(
                n_nodes,
                kernel_regularizer=_regularizers.L2(reg_L2_weight), 
                activation=hidden_layer_activation,
                kernel_initializer=initializer
            )(layer)
            layer = _layers.Dropout(dropout_rate)(tlayer)
            if(do_batch_normalization): layer = _layers.BatchNormalization()(layer)        
        
        layer = output_layer_func(layer)
        model = Model(
            inputs, layer,
            self.training_set_properties,
            self.training_set_labels,
            self.validation_set_properties,
            self.validation_set_labels
        )

        if(reg_WD_rate == None):
            model.compile(
                loss=loss,
                optimizer=_tf.optimizers.Adam(learning_rate=Adam_learning_rate),
            )
        elif(reg_L2_weight > 0):
            raise RuntimeError("Cannot have both L2 regularization and weight decay")
        else:
            model.compile(
                loss=loss,
                optimizer=_tf.optimizers.AdamW(
                    learning_rate=Adam_learning_rate,
                    weight_decay=reg_WD_rate,
                ),
            )

        if("verbose" in kwargs and kwargs["verbose"]>0):
            print(model.summary())
        
        return model
        

    def build_NN(
            self,
            target_variable_number=1,
            output_layer_activation='linear',
            initializer='he_normal',
            loss=_tf.losses.MeanSquaredError(),
            **kwargs
    ):
        """define and compile a fully connected NN.

        Keyword arguments:
        nodes_hidden_layers -- list of number of nodes. List lenght determines
                               the number of hidden layers
        nodes_output -- the number of output nodes
        reg_L2_weight -- value of all L2 regularization weights
        dropout_rate -- value of all dropout rates
        hidden_layer_activation
        output_layer_activation
        initializer
        Adam_learning_rate
        loss -- loss function
        """
        
        output_layer_func = _layers.Dense(
            target_variable_number,
            activation=output_layer_activation, # Notice no regularization
            kernel_initializer=initializer
        )

        return self.__build_network_with_given_output_layer(
            output_layer_func=output_layer_func,
            initializer=initializer,
            loss=loss,
            **kwargs
        )


    def build_MDN(
            self,
            output_components=1,
            target_variable_number=1,
            has_correlations=False,
            has_initializer_v2=False,
            output_mean_activation='elu_plus_one',
            initializer='he_normal',
            sigma_norms=1.,
            loss=losses.mse_means_and_sigmas_uncorrelated,
            **kwargs
    ):
        """define and compile a Mixture density NN.

        Keyword arguments:
        nodes_hidden_layers -- list of number of nodes. List lenght determines
                               the number of hidden layers
        output_components -- the number of components in the mixture
        reg_L2_weight -- value of all L2 regularization weights
        dropout_rate -- value of all dropout rates
        hidden_layer_activation
        output_layer_activation
        initializer
        Adam_learning_rate
        loss -- loss function
        """

        def output_layer_func(inputs):
            
            means = _layers.Dense(output_components*target_variable_number,
                                  activation=output_mean_activation,
                                  kernel_initializer=initializer)(inputs)
            
            _tf.keras.backend.print_tensor(inputs)
            
            if(has_correlations and output_components > 1):
                raise NotImplementedError()
                
            if(has_correlations):
                if(has_initializer_v2):
                    print("using initializer v2")
                    def initializer_pseudo_sigmas(shape, dtype=None):
                        return ( (0.5*_tf.ones(shape)+_tf.random.uniform(shape))/ _np.sqrt(shape[0]) / 10.
                                 * _tf.cast(_tfpmath.fill_triangular_inverse(_np.identity(target_variable_number)), dtype=dtype) )
                else:
                    def initializer_pseudo_sigmas(shape, dtype=None):
                        return ( _tf.math.sqrt((_tf.ones(shape)+_tf.random.uniform(shape))/ shape[0] / 100.)
                                 * _tf.cast(_tfpmath.fill_triangular_inverse(_np.identity(target_variable_number)), dtype=dtype) )
                
                pseudo_sigmas = _layers.Dense(target_variable_number*(target_variable_number+1)/2, 
                                        activation="linear",
                                        kernel_initializer=initializer_pseudo_sigmas)(inputs)   
                output_layer = _layers.Concatenate()([means, pseudo_sigmas])
                return output_layer
            
            sigmas = _tf.multiply(
                _layers.Dense(output_components*target_variable_number, 
                                   activation="elu_plus_one",
                                   kernel_initializer=initializer)(inputs),
                _tf.constant(sigma_norms, dtype=_tf.float32)
            )

            if(output_components > 1):
                weights = _layers.Dense(output_components*target_variable_number, 
                                        activation="softmax",
                                        kernel_initializer=initializer)(inputs)
                output_layer = _layers.Concatenate()([weights, means, sigmas])
                return output_layer

            #no correlations, single component
            output_layer = _layers.Concatenate()([means, sigmas])
            return output_layer
            
            
            
        return self.__build_network_with_given_output_layer(
            output_layer_func=output_layer_func,
            initializer=initializer,
            loss=loss,
            **kwargs
        )


def hypermodel_DO_WD_NodesLayers_LR(
    NN,
    DO_min=0.3, DO_max=0.7,
    WD_min=1.e-5, WD_max=1.e-3,
    N_nodes_min=8, N_nodes_max=2048,
    N_layers_min=1, N_layers_max=8,
    LR_min=1.e-5, LR_max=1.e-2,
    **kwargs,
):
    class MyHyperModel(_kt.HyperModel):
        def build(self, hp):
            # Tune floats
            hp_dropout_rate = hp.Float('dropout_rate', DO_min, DO_max)
            hp_reg_WD_weight = hp.Float(
                'reg_WD_rate', 
                min_value=WD_min, max_value=WD_max, 
                sampling="log"
            )
            # Tune from list

            #hp_architecture_index = hp.Choice('architecture_index' , range(len(architectures)))
            N_layers = hp.Int('N_layers', min_value=N_layers_min, max_value=N_layers_max)
            N_nodes = hp.Int('N_nodes', min_value=N_nodes_min, max_value=N_nodes_max)
            
            hp_architecture=[]
            for i in range(N_layers):
                hp_architecture.append(N_nodes)

            model = NN.build_MDN(
                nodes_hidden_layers=hp_architecture,
                reg_WD_rate=hp_reg_WD_weight,
                dropout_rate=hp_dropout_rate,
                **kwargs
            )

            return model

        def fit(self, hp, model, *args, callbacks=[], **kwargs):
            hp_base_learning_rate = hp.Float('base_learning_rate', min_value=LR_min, max_value=LR_max, sampling="log")

            clr_triangular = CyclicLR(#mode='exp_range',
                base_lr=hp_base_learning_rate,
                max_lr=hp_base_learning_rate*4.,
                step_size=3*4, # recommended (2-8) x (training iterations in epoch)
                gamma=0.99994
            )

            return  model.fit(
                *args,
                callbacks=callbacks+[clr_triangular],
                 **kwargs
            )
    return MyHyperModel()

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.
