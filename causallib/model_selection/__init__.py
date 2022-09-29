from sklearn.model_selection import GridSearchCV as skGridSearchCV
from sklearn.model_selection import RandomizedSearchCV as skRandomizedSearchCV

from .search import causalize_searcher
from .split import TreatmentOutcomeStratifiedKFold
from .split import TreatmentStratifiedKFold

GridSearchCV = causalize_searcher(skGridSearchCV)
RandomizedSearchCV = causalize_searcher(skRandomizedSearchCV)
