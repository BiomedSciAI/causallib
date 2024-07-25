
class ColumnNameChangeWarning(UserWarning):
    """Warning that causallib renamed input name
    to ensure all columns are of a single type
    so scikit-learn>=1.2.0 is happy.

    See array validation:
    https://github.com/scikit-learn/scikit-learn/blob/8133ecaacca77f06a8c4c560f5dbbfd654f1990f/sklearn/utils/validation.py#L2271-L2280"""
    pass
