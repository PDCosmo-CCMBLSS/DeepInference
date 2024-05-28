from sklearn.model_selection import train_test_split as _train_test_split

def train_val_test_split(
        properties,
        labels,
        val_fraction,
        test_fraction,
        random_state=1,
):
    test_size = round(len(properties) * test_fraction)
    val_size = round(len(properties) * val_fraction)

    train_prop, test_prop, train_lab, test_lab = _train_test_split(
          properties, labels, 
          test_size=test_size, random_state=random_state)

    train_prop, val_prop, train_lab, val_lab = _train_test_split(
          train_prop, train_lab,
          test_size=val_size, random_state=random_state)
    
    return train_prop, val_prop, test_prop, train_lab, val_lab, test_lab


def get_data_dict(
        data,
        first_feature_index,
        val_fraction = 0.2,
        test_fraction = 0.2,
        random_state = 1,
):    
    """returns a dictionary with the input data divided in input and target variables.
    Each is further divided in train validation and test set
    
    Arguments:
    data : ndarray
        assumed to be 2d. The second index labels the variables. 
        The input variables follow the target variables.
    first_feature_index : int
        number of target variables
    val_fraction : float
        fraction of data to be used in the validation set. Default = 0.2.
    test_fraction : float
        fraction of data to be used in the test set. Default = 0.2.
    random_state : int
        seed of the rng. Default = 1
    """
    
    data_dict = {
        "train" : {},
        "val" : {},
        "test" : {},
    }
    
    lbl = data[:,0:first_feature_index].copy()
    ftr = data[:,first_feature_index:].copy()    
    
    data_dict["train"]["ftr"],\
    data_dict["val"]["ftr"],\
    data_dict["test"]["ftr"],\
    data_dict["train"]["lbl"],\
    data_dict["val"]["lbl"],\
    data_dict["test"]["lbl"] = \
    train_val_test_split(
        ftr, lbl,
        val_fraction,
        test_fraction,
        random_state,
    )
    
    return data_dict