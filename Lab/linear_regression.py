import numpy as np
from scipy import stats


class LinearRegression:
    """
    Multiple Linear Regression (OLS) with statistical inference.
    - Uses numpy for linear algebra 
    - Uses scipy.stats for distributions/tests
    """

    def __init__(self, confidence_level: float = 0.95, add_intercept: bool = True):
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(
                "confidence_level must be between 0 and 1 (exclusive).")
        self.confidence_level = confidence_level
        self.add_intercept = add_intercept

        # learned parameters
        self.beta_ = None
        self.feature_names_ = None

        # cached training stats
        self.n_ = None
        self.d_ = None
        self.p_ = None
        self.df_ = None
        self.sse_ = None
        self.sigma2_ = None
        self.sigma_ = None
        self.cov_beta_ = None
        self.y_mean_ = None
        self.syy_ = None
        self.ssr_ = None
        self.r2_ = None

    # ---------- helpers ----------
    @staticmethod
    def _as_2d(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 1D or 2D array-like.")
        return X

    @staticmethod
    def one_hot_encode(col, drop_first: bool = True):
        """
        One-hot encode a 1D categorical array of strings/objects. 
        Returns (encoded_matrix, categories)
        """
        col = np.asarray(col)
        if col.ndim != 1:
            raise ValueError("col must be 1D.")
        categories = np.unique(col)
        k = len(categories)
        if k <= 1:
            return np.zeros((col.shape[0], 0)), categories

        start = 1 if drop_first else 0
        used = categories[start:]
        out = np.zeros((col.shape[0], len(used)), dtype=float)
        for j, cat in enumerate(used):
            out[:, j] = (col == cat).astype(float)
        return out, categories

    def _build_design_matrix(self, X, categorical_cols=None, drop_first: bool = True):
        """
        Build numeric design matrix with optional categorical columns.
        - X: 2D array-like, can be dtype=object if it mixes types
        - categorical_cols: list of column indices that are categorical
        """
        X = self._as_2d(X)

        if categorical_cols is None:
            X_num = X.astype(float)
        else:
            categorical_cols = set(categorical_cols)
            num_parts = []
            cat_parts = []

            for j in range(X.shape[1]):
                if j in categorical_cols:
                    enc, _ = self.one_hot_encode(
                        X[:, j], drop_first=drop_first)
                    cat_parts.append(enc)
                else:
                    num_parts.append(X[:, j].astype(float).reshape(-1, 1))

            X_num = np.hstack(num_parts) if num_parts else np.zeros(
                (X.shape[0], 0))
            X_cat = np.hstack(cat_parts) if cat_parts else np.zeros(
                (X.shape[0], 0))
            X_num = np.hstack([X_num, X_cat])

        if self.add_intercept:
            ones = np.ones((X_num.shape[0], 1), dtype=float)
            X_design = np.hstack([ones, X_num])
        else:
            X_design = X_num

        return X_design

    # ---------- core API ----------
    def fit(self, X, y, categorical_cols=None, drop_first: bool = True, feature_names=None):
        """
        Fit OLS regression.
        X: (n, d_raw)
        y: (n,) or (n,1)
        categorical_cols: indices in raw X that are categorical
        """
        y = np.asarray(y).reshape(-1)
        X_design = self._build_design_matrix(
            X, categorical_cols=categorical_cols, drop_first=drop_first)

        n = X_design.shape[0]
        p = X_design.shape[1]
        if y.shape[0] != n:
            raise ValueError("X and y must have tha same number of raws.")

        # d is number of features excluding intercept
        d = p - 1 if self.add_intercept else p

        # OLS solution : beta = pinv(X^T X) X^T y
        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ (X_design.T @ y)

        # predictions and residuals
        y_hat = X_design @ beta
        resid = y - y_hat

        # SSE, sigma**2
        sse = float(resid.T @ resid)

        if self.add_intercept:
            df = n - d - 1
        else:
            df = n - d
        if df <= 0:
            raise ValueError(
                "Not enough degrees of freedom (n too small vs number of parameters).")

        sigma2 = sse / df
        sigma = float(np.sqrt(sigma2))

        # total variation in y
        y_mean = float(np.mean(y))
        syy = float(np.sum((y - y_mean) ** 2))
        ssr = syy - sse
        r2 = float(ssr / syy) if syy > 0 else np.nan

        # store
        self.beta_ = beta
        self.n_ = n
        self.d_ = d
        self.p_ = p
        self.df_ = df
        self.sse_ = sse
        self.sigma2_ = sigma2
        self.sigma_ = sigma
        self.cov_beta_ = XtX_inv
        self.y_mean_ = y_mean
        self.syy_ = syy
        self.ssr_ = self.syy_ - self.sse_
        self.r2_ = r2
        self.feature_names_ = feature_names

        return self

    def predict(self, X, categorical_cols=None, drop_first: bool = True):
        if self.beta_ is None:
            raise RuntimeError("Model is not fitted.")
        X_design = self._build_design_matrix(
            X, categorical_cols=categorical_cols, drop_first=drop_first)
        return X_design @ self.beta_

    # ---------- required quantitative stats ----------
    @property
    def n(self):
        return self.n_

    @property
    def d(self):
        return self.d_

    def sample_variance(self):
        return self.sigma2_

    def sample_std(self):
        return self.sigma_

    def mse(self):
        if self.n_ is None:
            return None
        return self.sse_ / self.n_

    def rmse(self):
        m = self.mse()
        return float(np.sqrt(m)) if m is not None else None

    # ---------- significance, R^2, tests, CI, pearson, confidence level ----------
    def r2(self):
        return self.r2_

    def regression_significance(self):
        """
        F-test for overall regression significance:
        F = (SSR/d) / sigma^2
        df1 = d, df2 = n - d - 1 (with intercept)
        returns dict with F and p_value
        """
        if self.beta_ is None:
            raise RuntimeError("Model is not fitted.")
        if self.d_ <= 0:
            return {"F": np.nan, "P_value": np.nan}

        F = (self.ssr_ / self.d_) / self.sigma2_
        p_value = float(stats.f.sf(F, self.d_, self.df_))
        return {"F": float(F), "df1": int(self.d_), "df2": int(self.df_), "p_value": p_value}

    def coef_tests(self):
        """
        t-tests for individual coefficients.
        returns dict with t_stats, p_value, std_errors
        """
        if self.beta_ is None:
            raise RuntimeError("Model is not fitted.")
        se = np.sqrt(np.diag(self.cov_beta_) * self.sigma2_)
        t_stats = self.beta_ / se
        p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=self.df_)

        return {"beta": self.beta_.copy(), "std_error": se, "t": t_stats, "p_value": p_vals, "df": int(self.df_), }

    def confidence_intervals(self, confidence_level=None):
        """
        CI for each coefficient: beta_i -+ t_crit * sigma * sqrt(C_ii)
        """
        if self.beta_ is None:
            raise RuntimeError("Model is not fitted.")
        cl = self.confidence_level if confidence_level is None else float(
            confidence_level)
        if not (0.0 < cl < 1.0):
            raise ValueError(
                "confidence_level must be between 0 and 1 (exclusive).")

        alpha = 1.0 - cl
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=self.df_))

        se = np.sqrt(np.diag(self.cov_beta_) * self.sigma2_)
        lo = self.beta_ - t_crit * se
        hi = self.beta_ + t_crit * se
        return {"confidence_level": cl, "lower": lo, "upper": hi, "t_crit": t_crit, "df": int(self.df_)}

    def pearson_matrix(self, X_raw, categorical_cols=None, drop_first: bool = True):
        """
        Pearson correlation between all pairs of FEATURES (not including intercept).
        For categorical columns, it uses the one-hot expanded numeric matrix.
        Returns (R, P) where:
          R[i,j] = Pearson r
          P[i,j] = p-value
        """
        X_design = self._build_design_matrix(
            X_raw, categorical_cols=categorical_cols, drop_first=drop_first)

        # remove intercept column if present
        X_feat = X_design[:, 1:] if self.add_intercept else X_design
        m = X_feat.shape[1]
        R = np.eye(m, dtype=float)
        P = np.zeros((m, m), dtype=float)

        for i in range(m):
            for j in range(i + 1, m):
                r, p = stats.pearsonr(X_feat[:, i], X_feat[:, j])
                R[i, j] = R[j, i] = float(r)
                P[i, j] = P[j, i] = float(p)

        return R, P
