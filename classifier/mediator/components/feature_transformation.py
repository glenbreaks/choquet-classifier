from sklearn.preprocessing import QuantileTransformer

#TODO n_quantiles parameter
#TODO make feature transformation callable
class FeatureTransformation:
    """Feature Transformation

        Using Quantile Transformation Scaler from scikit-learn


    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data, where n_samples is the number of samples and
        n_features is the number of features.
    """

    def __init__(self, X):
        self.normalized = self._initialize(X)

    def _initialize(self, X):
        """Initialize the function.
                Create an array of shape (1, n_features), where n_features is
                the number of features. Each index stores an instance of the
                class _FeatureTransformationComponent describing a component f_i
                Parameters:
                ----------
                X : array-like of shape (n_samples, n_features)
                    Input data, where n_samples is the number of samples and
                    n_features is the number of features.
                Returns:
                ----------
                array : ndarray of shape (, n_features)
                    Array containing a _FeatureTransformationComponent f_i for
                    each feature.
                """
        return QuantileTransformer().fit_transform(X)

    def __getitem__(self, i):
        # use the item operator to access the feature transformation
        # component f_i
        return self.normalized[:, i]

    #def __call__(self, x):
        # use the call operator to compute a normalized value for x
