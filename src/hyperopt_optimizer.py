
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.base import clone
from sklearn.metrics import accuracy_score

def hyperopt_objective(params, model, X_train, y_train, X_test, y_test):
    clf = clone(model).set_params(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return {'loss': -accuracy, 'status': STATUS_OK}

def optimize_hyperopt(model, param_space, X_train, y_train, X_test, y_test, max_evals=10):
    trials = Trials()
    best_params = fmin(
        fn=lambda params: hyperopt_objective(params, model, X_train, y_train, X_test, y_test),
        space=param_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )
    return best_params