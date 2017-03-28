from sklearn import mixture, linear_model
import numpy as np

def initialise_mixture_model(n_fishes = None):
    if n_fishes == None:
        n_fishes = 2 #TODO: replace with estimate
    mxModel = mixture.GaussianMixture(n_components=n_fishes, covariance_type='full', tol=1e-3, max_iter=500)
    return mxModel


def get_parameters_for_heatmap(heatmap, mxModel):
    mxModel.fit(heatmap)
    w = mixture.GaussianMixture.weights_
    sigmas = mixture.GaussianMixture.covariances_ #TODO: eigendecompose this
    return w, sigmas


def initialise_classifier():
    heatMan = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    return heatMan


def train_classifier(model, summaries, labels):
    model.fit(summaries, labels)
    return model


