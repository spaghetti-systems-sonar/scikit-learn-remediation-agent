"""Utilities to discover scikit-learn objects."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path

_MODULE_TO_IGNORE = {
    "tests",
    "externals",
    "setup",
    "conftest",
    "experimental",
    "estimator_checks",
}

_SKLEARN_PREFIX = "sklearn."


def _should_ignore_module(module_name):
    """Check whether a module should be ignored during discovery."""
    module_parts = module_name.split(".")
    if any(part in _MODULE_TO_IGNORE for part in module_parts):
        return True
    return "._" in module_name


def _iter_sklearn_modules():
    """Yield imported sklearn modules, skipping ignored ones."""
    from sklearn.utils._testing import ignore_warnings

    root = str(Path(__file__).parent.parent)  # sklearn package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(
            path=[root], prefix=_SKLEARN_PREFIX
        ):
            if _should_ignore_module(module_name):
                continue
            yield import_module(module_name)


def _is_abstract(cls):
    if not hasattr(cls, "__abstractmethods__"):
        return False
    if not len(cls.__abstractmethods__):
        return False
    return True


def _normalize_type_filter(type_filter):
    """Normalize type_filter to a list and return a copy."""
    if not isinstance(type_filter, list):
        return [type_filter]
    return list(type_filter)


def _filter_estimators_by_type(estimators, type_filter, filters):
    type_filter = _normalize_type_filter(type_filter)
    filtered_estimators = []
    for name, mixin in filters.items():
        if name in type_filter:
            type_filter.remove(name)
            filtered_estimators.extend(
                [est for est in estimators if issubclass(est[1], mixin)]
            )
    if type_filter:
        raise ValueError(
            "Parameter type_filter must be 'classifier', "
            "'regressor', 'transformer', 'cluster' or "
            "None, got"
            f" {type_filter!r}."
        )
    return filtered_estimators


def _public_classes_in_module(module):
    """Return public classes defined in a single module."""
    return [
        (name, cls)
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if not name.startswith("_")
    ]


def _public_displays_in_module(module):
    """Return public display classes defined in a single module."""
    return [
        (name, cls)
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if not name.startswith("_") and name.endswith("Display")
    ]


def _collect_public_classes():
    """Walk sklearn packages and collect all public classes."""
    all_classes = []
    for module in _iter_sklearn_modules():
        all_classes.extend(_public_classes_in_module(module))
    return set(all_classes)


def _collect_non_abstract_estimators(all_classes, base_cls):
    """Filter public classes to non-abstract estimators."""
    estimators = [
        c
        for c in all_classes
        if issubclass(c[1], base_cls) and c[0] != "BaseEstimator"
    ]
    return [c for c in estimators if not _is_abstract(c[1])]


def all_estimators(type_filter=None):
    """Get a list of all estimators from `sklearn`.

    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.

    Parameters
    ----------
    type_filter : {"classifier", "regressor", "cluster", "transformer"} \
            or list of such str, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'cluster' and 'transformer' to get
        estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from sklearn.utils.discovery import all_estimators
    >>> estimators = all_estimators()
    >>> type(estimators)
    <class 'list'>
    >>> type(estimators[0])
    <class 'tuple'>
    >>> estimators[:2]
    [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>),
     ('AdaBoostClassifier',
      <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)]
    >>> classifiers = all_estimators(type_filter="classifier")
    >>> classifiers[:2]
    [('AdaBoostClassifier',
      <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),
     ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>)]
    >>> regressors = all_estimators(type_filter="regressor")
    >>> regressors[:2]
    [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>),
     ('AdaBoostRegressor',
      <class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>)]
    >>> both = all_estimators(type_filter=["classifier", "regressor"])
    >>> both[:2]
    [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>),
     ('AdaBoostClassifier',
      <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)]
    """
    # lazy import to avoid circular imports from sklearn.base
    from sklearn.base import (
        BaseEstimator,
        ClassifierMixin,
        ClusterMixin,
        RegressorMixin,
        TransformerMixin,
    )

    all_classes = _collect_public_classes()
    estimators = _collect_non_abstract_estimators(all_classes, BaseEstimator)

    if type_filter is not None:
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            "cluster": ClusterMixin,
        }
        estimators = _filter_estimators_by_type(
            estimators, type_filter, filters
        )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


def all_displays():
    """Get a list of all displays from `sklearn`.

    Returns
    -------
    displays : list of tuples
        List of (name, class), where ``name`` is the display class name as
        string and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from sklearn.utils.discovery import all_displays
    >>> displays = all_displays()
    >>> displays[0]
    ('CalibrationDisplay', <class 'sklearn.calibration.CalibrationDisplay'>)
    """
    all_classes = []
    for module in _iter_sklearn_modules():
        all_classes.extend(_public_displays_in_module(module))

    return sorted(set(all_classes), key=itemgetter(0))


def _is_checked_function(item):
    if not inspect.isfunction(item):
        return False

    if item.__name__.startswith("_"):
        return False

    mod = item.__module__
    if not mod.startswith(_SKLEARN_PREFIX) or mod.endswith("estimator_checks"):
        return False

    return True


def _public_functions_in_module(module):
    """Return public functions defined in a single module."""
    return [
        (func.__name__, func)
        for name, func in inspect.getmembers(module, _is_checked_function)
        if not name.startswith("_")
    ]


def all_functions():
    """Get a list of all functions from `sklearn`.

    Returns
    -------
    functions : list of tuples
        List of (name, function), where ``name`` is the function name as
        string and ``function`` is the actual function.

    Examples
    --------
    >>> from sklearn.utils.discovery import all_functions
    >>> functions = all_functions()
    >>> name, function = functions[0]
    >>> name
    'accuracy_score'
    """
    all_funcs = []
    for module in _iter_sklearn_modules():
        all_funcs.extend(_public_functions_in_module(module))

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(all_funcs), key=itemgetter(0))
