# Implements Shared Sparsity Confounder Selection suggested by:
# https://arxiv.org/abs/2011.01979

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from causallib.preprocessing.confounder_selection import _BaseConfounderSelection

__all__ = ["SharedSparsityConfounderSelection"]


class MCPSelector:
    # TODO: transpose `theta` and rename it `coef_` to align with sklearn models
    def __init__(self, lmda="auto", alpha=1, step=0.1, max_iter=1000, tol=1e-3):
        """Constructor for MCPSelector. This class computes shared
        sparsity matrix using proximal gradient descent applied with
        MCP regularizer.

        Args:
            lmda (str|float): Parameter (>= 0) to control shape of MCP regularizer.
                The bigger the value the stronger the regularization.
                "auto" will auto-select good regularization value.
            alpha (float): Associated lambda parameter (>= 0) to control shape of MCP regularizer.
                The smaller the value the stronger the regularization.
            step (float): Step size for proximal gradient, equivalent of learning rate.
            max_iter (int): Maximum number of iterations of MCP proximal gradient.
            tol (float): Stopping criterion for MCP. If the normalized value of
                proximal gradient is less than tol then the algorithm is assumed
                to have converged.
        """
        super().__init__()
        self.alpha = alpha
        self.lmda = lmda
        self.step = step
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon_safe_division = 1e-6

    def _initialize_internal_params(self, X, a, y):
        treatments = list(a.unique())
        n_treatments = len(treatments)
        n_confounders = X.shape[1]
        if self.lmda == "auto":
            avg_num_samples = len(y) / n_treatments
            lmda = 0.2 * np.sqrt(n_treatments * np.log(n_confounders) / avg_num_samples)
        else:
            lmda = self.lmda

        assert self.alpha >= 0
        assert lmda >= 0

        corr_xx = np.zeros((n_confounders, n_confounders, n_treatments))
        corr_xa = np.zeros((n_confounders, n_treatments))
        for i, t in enumerate(treatments):
            u, v = X.loc[a == t].values.T, y.loc[a == t].values
            corr_xx[:, :, i] = (u @ u.T) / len(v)
            corr_xa[:, i] = (u @ v) / len(v)
        self.theta_ = np.zeros((n_confounders, n_treatments))
        self.corr_mats_ = {"corr_xx": corr_xx, "corr_xa": corr_xa}  # Correlation matrices
        self.n_confounders_ = n_confounders
        self.n_treatments_ = n_treatments
        self.lmda_ = lmda

    def _mcp_q_grad(self, t):
        if self.alpha == 0:
            return np.sign(t) * self.lmda_ * (-1)
        else:
            return (np.sign(t) * self.lmda_
                    * np.maximum(-1, - np.abs(t) / (self.lmda_ * self.alpha)))

    def _update_grad_op(self, theta):
        theta_new = np.zeros_like(theta)
        for j in range(self.n_treatments_):
            sub = (self.corr_mats_["corr_xx"][:, :, j] @ theta[:, j]
                   - self.corr_mats_["corr_xa"][:, j])
            theta_new[:, j] = theta[:, j] - self.step * sub
        norm_matrix = (np.linalg.norm(theta, axis=1, keepdims=True)
                       @ np.ones((1, self.n_treatments_)))
        eps = self.epsilon_safe_division
        theta_grad_op = theta_new - (self.step * (self._mcp_q_grad(norm_matrix))
                                     * theta / (norm_matrix + eps))
        return theta_grad_op

    def _update_proximal_op(self, theta):
        norm_theta = np.linalg.norm(theta, axis=1, keepdims=True)
        shrink = np.maximum(0, norm_theta - self.lmda_)
        theta_proximal_op = theta * ((shrink / norm_theta)
                                     @ np.ones((1, self.n_treatments_)))
        return theta_proximal_op

    def compute_shared_sparsity_matrix(self, X, a, y):
        self._initialize_internal_params(X, a, y)
        theta = self.theta_
        for i in range(self.max_iter):
            theta_prev = theta
            theta_grad_op = self._update_grad_op(theta_prev)
            theta = self._update_proximal_op(theta_grad_op)

            convergence_criteria = (
                    np.linalg.norm(theta - theta_prev) /
                    (np.linalg.norm(theta) + self.epsilon_safe_division)
            )
            has_converged = convergence_criteria <= self.tol
            if has_converged:
                break
        else:
            warnings.warn("Shared sparsity did not converge in maximum iterations.", ConvergenceWarning)
        self.theta_ = theta
        return self.theta_


class SharedSparsityConfounderSelection(_BaseConfounderSelection):
    """Class to select confounders by first applying shared sparsity method.
    Method by Greenewald, Katz-Rogozhnikov, and Shanmugam: https://arxiv.org/abs/2011.01979
    """

    def __init__(self, mcp_lambda="auto", mcp_alpha=1, step=0.1, max_iter=1000, tol=1e-3, threshold=1e-6,
                 importance_getter=None, covariates=None):
        """Constructor for SharedSparsityConfounderSelection

        Args:
            mcp_lambda (str|float): Parameter (>= 0) to control shape of MCP regularizer.
                The bigger the value the stronger the regularization.
                "auto" will auto-select good regularization value.
            mcp_alpha (float): Associated lambda parameter (>= 0) to control shape of MCP regularizer.
                The smaller the value the stronger the regularization.
            step (float): Step size for proximal gradient, equivalent of learning rate.
            max_iter (int): Maximum number of iterations of MCP proximal gradient.
            tol (float): Stopping criterion for MCP. If the normalized value of
                proximal gradient is less than tol then the algorithm is assumed
                to have converged.
            threshold (float): Only if the importance of a confounder exceeds
                threshold for all values of treatments, then the confounder
                is retained by transform() call.
            importance_getter: IGNORED.
            covariates (list | np.ndarray): Specifying a subset of columns to perform selection on.
                Columns in `X` but not in `covariates` will be included after `transform`
                no matter the selection.
                Can be either a list of column names, or an array of boolean indicators length of `X`,
                or anything compatible with pandas `loc` function for columns.
                if `None` then all columns are participating in the selection process.
                This is similar to using sklearn's `ColumnTransformer` or `make_column_selector`.
        """
        super().__init__(importance_getter=importance_getter, covariates=covariates)
        self.step = step
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold
        self.selector_ = MCPSelector(lmda=mcp_lambda, alpha=mcp_alpha, step=step, max_iter=max_iter, tol=tol)

        self.importance_getter = lambda e: e.selector_.theta_.transpose()  # Shape to behave like sklearn linear_model
        # self.importance_getter = "selector_.theta_"

    @_BaseConfounderSelection._filter_covariates
    def fit(self, X, a, y):
        # compute_shared_sparsity_matrix() below should return
        # a matrix of shape (n_confounders x n_treatments).
        # The confounders to be retained in transform() corresponds to those rows
        # which have significant values across all the columns.
        theta = self.selector_.compute_shared_sparsity_matrix(X, a, y)
        support = (np.abs(theta) >= self.threshold).all(axis=1)
        if not support.any():
            warnings.warn("Shared sparsity selected zero features. Ignoring and selecting all features.")
            self.support_ = np.ones(X.shape[1], dtype=bool)
        else:
            self.support_ = support
        self.n_features_ = sum(self.support_)
        return self

    # @_BaseConfounderSelection._filter_and_re_add_covariates
    # def transform(self, X, a=None):
    #     X = X.loc[:, self.get_support()]
    #     return X

    def _get_support_mask(self):
        return self.support_
