import tensorflow as _tf

def mse_means_and_sigmas_uncorrelated(y_true, y_pred):
    """Mean squared error for a moment network.

    Mean squared error for the means and standard deviations according to the moment network prescription.
    The mse's of mean and sigma for each target parameter are averaged, according to eq. (26) of ([2109.10915])[https://arxiv.org/abs/2109.10915].

    Parameters
    ----------
    y_true : tensor
        True value of the target parameters.
    y_pred : tensor
        Tensor with same shape[0] as y_true, and twice the shape[1] of y_true.
        The first half of the parameters along the second axis are the predicted means of the posterior,
        while the latter half are the predicted standard deviations.

    Returns
    -------
    float
        Loss

    """
    means_pred, sigmas_pred = _tf.split(y_pred, num_or_size_splits=2, axis=1)
    
    y_true = _tf.cast(y_true, dtype=y_pred.dtype)
    
    squared_differences = _tf.math.square(y_true - means_pred)
    sigmas2_sigma = _tf.math.reduce_mean(_tf.math.square(squared_differences - _tf.math.square(sigmas_pred)), 0)
    sigmas2 = _tf.math.reduce_mean(squared_differences, 0) 
    
    loss = _tf.math.reduce_mean(sigmas2 + sigmas2_sigma)

    return loss


def logmse_means_and_sigmas_uncorrelated(y_true, y_pred):
    means_pred, sigmas_pred = _tf.split(y_pred, num_or_size_splits=2, axis=1)
    
    y_true = _tf.cast(y_true, dtype=y_pred.dtype)
    
    squared_differences = _tf.math.square(y_true - means_pred)
    sigmas2_sigma = _tf.math.reduce_mean(_tf.math.square(squared_differences - _tf.math.square(sigmas_pred)), 0)
    sigmas2 = _tf.math.reduce_mean(squared_differences, 0) 
    
    loss = _tf.math.reduce_mean(_tf.math.log(sigmas2) + _tf.math.log(sigmas2_sigma))

    return loss


def mse_means_and_sigmas_correlated(y_true, y_pred):
    num_of_parameters=_tf.cast((_tf.math.sqrt(_tf.cast(8*y_pred.shape[-1]+9, dtype=_tf.float16))-3)/2, dtype=_tf.int32)
    means_pred, pseudo_sigmas_pred = _tf.split(y_pred,
                                              num_or_size_splits=[
                                                  num_of_parameters,
                                                  _tf.cast(num_of_parameters*(num_of_parameters+1)/2, dtype=_tf.int32)
                                              ], axis=-1)

    L = _tfpmath.fill_triangular(pseudo_sigmas_pred)
    
    dist = _distributions.MultivariateNormalTriL(
        loc=means_pred,
        scale_tril=L,
    )
    
    loglkl = dist.log_prob(y_true)#_tf.transpose(y_true))
    
    return -_tf.math.reduce_mean(loglkl)


def minus_loglikelihood_normal(y_true, params):
    y_true = _tf.cast(y_true, dtype=params.dtype) 
    
    dist = _distributions.Normal(
        params[:,0],
        params[:,1],
    )
    
    loglkl = dist.log_prob(_tf.transpose(y_true))
    
    return -_tf.math.reduce_mean(loglkl)


def minus_loglikelihood_Poisson(y_true, params):
    y_true = _tf.cast(y_true, dtype=params.dtype) 
    
    dist = _distributions.Poisson(
        params[:,0],
    )
    
    loglkl = dist.log_prob(_tf.transpose(y_true))
    
    return -_tf.math.reduce_mean(loglkl)


def minus_loglikelihood_lognormal(y_true, params):
    y_true = _tf.cast(y_true, dtype=params.dtype) 
    
    dist = _distributions.LogNormal(
        params[:,0],
        params[:,1],
    )
    
    loglkl = dist.log_prob(_tf.transpose(y_true))
    
    return -_tf.math.reduce_mean(loglkl)


def minus_loglikelihood_distribution_mixture(distribution):
    
    def loss(y_true, params):
        components=_tf.Tensor.get_shape(params)[1]/3

        weights=params[:,:components]
        means=params[:,components:2*components]
        sigmas=params[:,2*components:]   

        dist = _distributions.MixtureSameFamily(
            mixture_distribution=_distributions.Categorical(probs=weights),
            components_distribution=distribution(means,
                                                 sigmas),
        )

        loglkl = dist.log_prob(y_true[:,0])
        return -_tf.math.reduce_mean(loglkl)
    
    return loss


def minus_loglikelihood_lognormal_mixture(y_true, params):
    return minus_loglikelihood_distribution_mixture(_distributions.LogNormal)(y_true, params)


def minus_loglikelihood_normal_mixture(y_true, params):
    return minus_loglikelihood_distribution_mixture(_distributions.Normal)(y_true, params)
