from sklearn.model_selection import RandomizedSearchCV

def optimize_random_search(model, param_distributions, X_train, y_train, n_iter=10, cv=3, scoring='roc_auc'):
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_params_