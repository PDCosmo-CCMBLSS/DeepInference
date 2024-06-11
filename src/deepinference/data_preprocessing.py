from sklearn.model_selection import train_test_split as _train_test_split

def train_val_test_split(
        properties,
        labels,
        val_fraction,
        test_fraction,
        random_state=1,
):
    """Split data in train, validation and test set.

    Split data in train, validation and test set according to the fraction selected
    applying twice the scikit learn routine train_test_split.
    See the (train_test_split documentation)[https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html].

    Parameters
    ----------
    properties : indexable with same lenght/shape[0] as labels
        Tabular input features. Lenght/shape[0] is the number of data samples.
    labels : indexable with samel lenght/shape[0] as properties
        Tabular target variables.
    val_fraction : float
        Fraction of samples to be included in the validation set.
    test_fraction : float
        Fraction of samples to be included in the test set.
    random_state : int
        Control the shuffling applied to data before applying the split. Same value
        guarantees reproducible output.

    Returns
    -------
    list of length=6
        List of train, validation, and test properties, train, validation, and test labels.

    """
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

    Parameters
    ----------
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

    Returns
    -------
    dictionary
        Dictionary with 3 different items train, val, and test, further subdivided in 
        ftr (features) and lbl (labels)

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
