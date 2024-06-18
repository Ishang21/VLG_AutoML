from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def plot_roc_auc_curve(model, X_test, y_test, label):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.2f})')
    return auc_score


def calculate_cross_val_score(model, X_train, y_train):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)
    return cv_scores, mean_cv_score

def plot_learning_rate_distribution_curve(model, X_test, y_test, label):
    y_proba = model.predict_proba(X_test)
    plt.hist(y_proba[:, 1], bins=30, alpha=0.5, label=label)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
#     plt.show()



def plot_learning_curve(model, X, y, label, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label=f"Training score ({label})")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label=f"Cross-validation score ({label})")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.title(f'Learning Curve ({label})')
    
    
def evaluate_and_compare(models, X_train, y_train, X_test, y_test, labels, cv=5):
    plt.figure(figsize=(10, 7))
    
    for model, label in zip(models, labels):
        # ROC AUC
        auc_score = plot_roc_auc_curve(model, X_test, y_test, label)
        print(f"ROC AUC score ({label}):", auc_score)
        
        # Cross-validation scores
        cv_scores, mean_cv_score = calculate_cross_val_score(model, X_train, y_train)
        print(f"Cross-validation scores ({label}):", cv_scores)
        print(f"Mean CV score ({label}):", mean_cv_score)
        
        # Learning rate distribution curve
        plot_learning_rate_distribution_curve(model, X_test, y_test, label)
    
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC AUC Curves')
#     plt.legend(loc='best')
#     plt.show()
    
#     plt.figure(figsize=(10, 7))
#     plt.title('Learning Rate Distribution Curves')
#     plt.xlabel('Predicted Probability')
#     plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.show()

#     # Plot learning curves
#     for model, label in zip(models, labels):
#         plt.figure(figsize=(10, 7))
#         plot_learning_curve(model, X_train, y_train, label, cv=cv)
#         plt.show()
