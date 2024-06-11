import matplotlib.pyplot as plt
import numpy as np

import keras_tuner as _kt

def check_predictions(
    trueY,
    predicY,
    predicE,
    label='quantity [some units]',
    title=None,
    color_error_dependence=None,
):
    """Creates plots to compare true and predicted values.

    Creates a figure with three panels: predicted vs true parameter values, predicted errors vs true parameter
    values, and bias of the predictions vs true parameter values.
    
    Parameters
    ----------
    trueY : ndarray
        True labels values, has the same length as predicY and predicE
    predicY : ndarray
        Predicted mean of the label posterior.
    predicE : ndarray
        Predicted standard deviation of the label posterior.
    label : string
        xlabel and ylabel are set to "True "/"Predicted "+label
    title : string or None
        if not None, sets figure title
    color_error_dependence : ndarray or None
        true (or predicted) value of another parameter (currently
        Omega_m is hardcoded) correlated with the error.
        If None, the default color is used.
        [TODO remove hardcoded assumptions]

    """
    fig, ax = plt.subplots(ncols=3,sharex=True,figsize=(9,2.8), dpi=200)#6.4,2.8 #6.4,4.8

    ax[0].errorbar(
        x=trueY[:], y=predicY,
        yerr=predicE,
        elinewidth=0.5,
        linewidth=0,
    )
    extremes = [np.min([trueY, predicY]),np.max([trueY, predicY])]    
    ax[0].set_xlabel('True '+label)
    ax[0].set_ylabel('Predicted '+label)
    ax[0].plot(extremes, extremes, c='k')
    ax[0].set_xlim(extremes[0], extremes[1])
    ax[0].set_ylim(extremes[0], extremes[1])
    ax[0].set_aspect('equal', adjustable='box')
    
    ymean = np.mean(trueY)
    R2 = 1.-np.sum((trueY-predicY)**2) / np.sum((trueY-ymean)**2)
    
    ax[0].text(0.975, 0.025, r'$R^2$=%.2f'
               "\n"
               r"$\chi^2$=%.2f" %(R2, np.sum((trueY - predicY)**2/predicE**2)/(len(predicE)-2)),
               style='italic', transform=ax[0].transAxes,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2}, ha="right", va="bottom")
    
    if(color_error_dependence is not None):
        pcm = ax[1].scatter(
            trueY, 
            predicE, 
            c=color_error_dependence,
            marker=".",
            s=4, 
            alpha=1)
        cax = ax[1].inset_axes([0.6, 0.95, 0.4, 0.05])
        cbar = fig.colorbar(pcm, cax=cax, orientation='horizontal')
        cbar.set_label(r'$\Omega_m$', labelpad=-10.5, x=-.1)
    else:
        ax[1].plot(trueY, predicE, marker=".", lw=0, markersize=2, alpha=1)
    ax[1].set_xlabel('True '+label)
    ax[1].set_ylabel('Standard deviation')
    ax[1].text(0.975, 0.025, r'$\langle\sigma \rangle$=%.2f'
               "\n"
               r"RMSE=%.2f" %(np.mean(predicE), np.sqrt(np.mean((predicY-trueY)**2))), 
               style='italic', transform=ax[1].transAxes,
               bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
               ha="right", va="bottom")
    
    ax[2].grid(axis="y",alpha=0.5,ls="--")
    
    ax[2].plot(trueY, (predicY-trueY)/predicE, marker=".", lw=0, markersize=2, alpha=1)
    ax[2].set_xlabel('True '+label)
    ax[2].set_ylabel(r'Bias [$\sigma$]')
    ax[2].grid(axis="y",alpha=0.5,ls="--")
    ax[2].text(0.975, 0.025, r"$\langle {\rm bias} \rangle$=%.2f"
               "\n"
               r"$\langle |{\rm bias}| \rangle$=%.2f" % (np.mean((predicY-trueY)/predicE), np.mean(np.abs(predicY-trueY)/predicE)), 
               style='italic', transform=ax[2].transAxes,
               bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
               ha="right", va="bottom")
    plt.tight_layout()
    
    if title is not None:
        plt.subplots_adjust(left=0.1, right=0.975, top=0.9, bottom=0.2)
        plt.suptitle(title)#, fontdict={'horizontalalignment': "center"})
    else:
        plt.subplots_adjust(left=0.05, right=0.975, top=0.975, bottom=0.2)
    fig.show()
    

def get_ranked_hyperparameters(
    directory, 
    model, 
    dset,
    loss_key, 
    num_trials=50
):
    tuner = _kt.BayesianOptimization(
        model,
        objective=_kt.Objective("val_loss", direction="min"),
        max_trials=20,
        directory=directory,
        project_name='tuning - '+loss_key+' - '+dset,
    )

    return tuner.get_best_hyperparameters(num_trials=num_trials)


def show_best_hyperparameters(ranked_hypars, hypars_names, n_models):
    for i in range(n_models):
        for hp_name in hypars_names:
            is_float = isinstance(ranked_hypars[i][hp_name], float)
            if(is_float and ranked_hypars[i][hp_name] < 1.e-2):
                print(hp_name, ("%.4e" %ranked_hypars[i][hp_name]) .ljust(12), end="  ")
            elif(is_float):
                print(hp_name, ("%.4f" %ranked_hypars[i][hp_name]) .ljust(12), end="  ")
            else:
                print(hp_name, str(ranked_hypars[i][hp_name]) .ljust(12), end="  ")
        print()
        
        
def get_ranked_chi2s(
    directory,
    model,
    dset,
    means,
    sigmas,
    loss_key,
    num_shown_models,
):
    tuner = _kt.BayesianOptimization(
        model,
        objective=_kt.Objective("val_loss", direction="min"),
        max_trials=20,
        directory=directory,
        project_name='tuning - '+loss_key+' - '+dset,
    )
    
    models = tuner.get_best_models(num_models=num_shown_models)
    
    if(isinstance(models[0].validation_set_labels[0],float)):
        models[0].validation_set_labels = np.expand_dims(models[0].validation_set_labels, axis=1)
        
    target_var_number = len(models[0].validation_set_labels[0])
    chi2s = np.zeros((len(models), target_var_number))
   
    for i, model in enumerate(models):
        predictions = model.predict(models[0].validation_set_properties, verbose=0)
       
        for j in range(target_var_number):
            trueY = models[0].validation_set_labels[:,j]*sigmas[j]+means[j]
            predicY = predictions[:,j]*sigmas[j]+means[j]
            predicE = predictions[:,j+target_var_number]*sigmas[j]

            chi2s[i,j] = np.sum((trueY - predicY)**2/predicE**2)/(len(predicE)-2)

    return chi2s


def plot_chi2s(
    chi2s, 
    labels, 
    title=None
):
    for i in range(len(labels)):
        plt.plot(range(len(chi2s)), chi2s[:,i], label=labels[i])

    plt.ylim(0,5)
    plt.xlim(-0.2,20)
    plt.axhline(0.75, alpha=0.5, c="k", lw=0.5, dashes=[6,6])
    plt.axhline(1.25, alpha=0.5, c="k", lw=0.5, dashes=[6,6])
    plt.xlabel("model (rank)")
    plt.ylabel("$\chi^2$")
    plt.legend(frameon=False, ncol=5)
    if(title is not None):
        plt.title(title)

    plt.show()
    
    maxchi2 = np.max(np.abs(chi2s-1),axis=1)
    plt.plot(range(len(chi2s)), maxchi2)

    plt.ylim(0,5)
    plt.xlim(-0.2,40)
    plt.axhline(0.25, alpha=0.5, c="k", lw=0.5, dashes=[6,6])    
    plt.axhline(0.5, alpha=0.5, c="k", lw=0.5, dashes=[6,6])
    plt.xlabel("model (rank)")
    plt.ylabel("max$|\chi^2-1|$")
    if(title is not None):
        plt.title(title)

    plt.show()
    
    
def show_nth_best_model(
    directory, 
    model,
    dset,
    means,
    sigmas, 
    labels, 
    loss_key, 
    model_rank, 
    data, 
    dset_name=None,
    data_split = "test"
):
    
    tuner = _kt.BayesianOptimization(
        model,
        objective=_kt.Objective("val_loss", direction="min"),
        max_trials=20,
        directory=directory,
        project_name='tuning - '+loss_key+' - '+dset,
    )
    
    models = tuner.get_best_models(num_models=model_rank+1)
    hypars = tuner.get_best_hyperparameters(num_trials=model_rank+1)
    
    model = models[-1]
    hypar = hypars[-1]

    predictions = model.predict(data[data_split]["ftr"], verbose=0)
    target_var_number = len(data[data_split]["lbl"][0])

    j = 0
    for i in range(target_var_number):
        print(
            data[data_split]["lbl"][j][i]*sigmas[i]+means[i],
            " -> ", predictions[j][i]*sigmas[i]+means[i], 
            "+-",   predictions[j][target_var_number+i]*sigmas[i], 
            " (", (data[data_split]["lbl"][j][i]-predictions[j][i])/predictions[j][target_var_number+i], ")"
        )


    biases = np.zeros(target_var_number)
    num_biases_g1 = np.zeros(target_var_number)
    errors = np.zeros(target_var_number)
    for j in range(len(data[data_split]["lbl"])):
        for i in range(target_var_number):
            biases[i] += np.abs((data[data_split]["lbl"][j][i]-predictions[j][i])/predictions[j][target_var_number+i])
            errors[i] += predictions[j][target_var_number+i]*sigmas[i]
            num_biases_g1[i] += (np.abs((data[data_split]["lbl"][j][i]-predictions[j][i])/predictions[j][target_var_number+i]) > 1.)

    for i in range(target_var_number):
        biases[i] /= len(data[data_split]["lbl"])
        errors[i] /= len(data[data_split]["lbl"])

    model.evaluate(data[data_split]["ftr"], data[data_split]["lbl"])

    if(dset_name == None):
        analysis_key = (
            dset +
            " - DO %.2f"%(hypar['dropout_rate']) +
            " - WD %.2e"%(hypar['reg_WD_rate']) +
            " - LR %.2e"%(hypar['base_learning_rate']) +
            " - ARC %iNx%iL"%(hypar['N_nodes'], hypar['N_layers'])
        )
    else:
        analysis_key = dset_name

    for j in range(target_var_number):
        check_predictions(
            data[data_split]["lbl"][:,j]*sigmas[j]+means[j],
            predictions[:,j]*sigmas[j]+means[j], 
            predictions[:,j+target_var_number]*sigmas[j],
            label=labels[j],
            title=analysis_key)
        plt.show()
    print()