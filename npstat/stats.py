import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
import numpy as np


def mann_whitney_test(df, categorical_vars, numerical_variable):
    results = {}
    
    if isinstance(numerical_variable, str):
        numerical_variable = [numerical_variable]
    if isinstance(categorical_vars, str):
        categorical_vars = [categorical_vars]

    for cat_var in categorical_vars:
        for variable in numerical_variable:
            
            categories = df[cat_var].unique()
            
            if len(categories) == 2:
                group_1 = df[df[cat_var] == categories[0]][variable]
                group_2 = df[df[cat_var] == categories[1]][variable]
                
               
                u_statistic, p_value = scipy.stats.mannwhitneyu(group_1, group_2, alternative='two-sided')
                
                
                if cat_var not in results:
                    results[cat_var] = {}
                results[cat_var][variable] = {'U-statistic': u_statistic, 'p-value': p_value}


    results_df = pd.DataFrame({
        (cat_var, variable): results[cat_var].get(variable, {}) for cat_var in categorical_vars for variable in numerical_variable
    }).T


    def highlight_significant(val):
        color = 'background-color: yellow' if val < 0.05 else ''
        return color

    styled_results_df = results_df.style.map(highlight_significant, subset=pd.IndexSlice[:, 'p-value'])

    return styled_results_df


def kruskal_wallis_test(df, categorical_vars, numerical_variable):
    
    results = {}
    if isinstance(numerical_variable, str):
        numerical_variable = [numerical_variable]
    if isinstance(categorical_vars, str):
        categorical_vars = [categorical_vars]

    for cat_var in categorical_vars:
        categories = df[cat_var].unique()
        
        
        if len(categories) >= 3:
            for variable in numerical_variable:
                
                groups = [df[df[cat_var] == category][variable] for category in categories]
                
            
                h_statistic, p_value = scipy.stats.kruskal(*groups)
                
               
                if cat_var not in results:
                    results[cat_var] = {}
                results[cat_var][variable] = {'H-statistic': h_statistic, 'p-value': p_value}


    if results:
      
        results_df = pd.DataFrame({
            (cat_var, variable): results[cat_var].get(variable, {}) for cat_var in categorical_vars for variable in numerical_variable
        }).T


        def highlight_significant(val):
            color = 'background-color: yellow' if val < 0.05 else ''
            return color

       
        styled_results_df = results_df.style.map(highlight_significant, subset=pd.IndexSlice[:, 'p-value'])

        return styled_results_df
    else:

        return pd.DataFrame()



def anova(data, numerical_variable, categorical_variable):

    results = {}

    if isinstance(numerical_variable, str):
        numerical_variable = [numerical_variable]

    for variable in numerical_variable:
        groups = [data[data[categorical_variable] == category][variable] for category in data[categorical_variable].unique()]
        f, p_value = scipy.stats.f_oneway(*groups)

        results[variable] = {'F-Statistics': f, 'p-value': p_value}

    results_df = pd.DataFrame(results).T


    def highlight_significant(val):
        color = 'background-color: yellow' if val < 0.05 else ''
        
        return color

       
    styled_results_df = results_df.style.map(highlight_significant, subset=pd.IndexSlice[:, 'p-value'])

    return styled_results_df



def chi_square(data, cls_cats, test_cats):
    
    results = {}

    if isinstance(cls_cats, str):
        cls_cats = [cls_cats]
    if isinstance(test_cats, str):
        test_cats = [test_cats]
    
    for col in test_cats:
        
        results[col] = {}
  
        for i in cls_cats:
            contingency = pd.crosstab(data[i], data[col])
            chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency)
            results[col][i] = {'Degree of Freedom': dof, 'chi square': chi2, 'p-value': p}
        
    results_df = pd.DataFrame({
        (col, i): results[col].get(i, {}) for col in test_cats for i in cls_cats
        }).T

    def highlight_significant(val):
        color = 'background-color: yellow' if val < 0.05 else ''
        return color

    styled_results_df = results_df.style.map(highlight_significant, subset=pd.IndexSlice[:, 'p-value'])


    return styled_results_df


def FeaturesPlot(model, X_train, X_test, y_test):
    """
    Plots the feature importances and ROC curves for a trained classification model (binary or multiclass).

    Parameters:
    - model: Trained classifier with feature_importances_ and predict_proba
    - X_train: Training feature set (used for feature names)
    - X_test: Test feature set
    - y_test: True labels for the test set
    """
    # Feature importances
    importances = model.feature_importances_
    feature_names = X_train.columns
    indices = np.argsort(importances)[::-1]

    # Classes info
    classes = np.unique(y_test)
    n_classes = len(classes)
    
    y_score = model.predict_proba(X_test)

    # Plotting setup
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # --- Feature Importances Plot ---
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=ax[0])
    ax[0].set_title("Feature Importances")
    ax[0].set_xlabel("Importance")
    ax[0].set_ylabel("Feature")

    # --- ROC Curve Plot ---
    if n_classes == 2:
        
        positive_class = classes[1] 
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label=positive_class)
        roc_auc = auc(fpr, tpr)
        ax[1].plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")

    else:
        # Multiclass case
        y_test_bin = label_binarize(y_test, classes=classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        for i in range(n_classes):
            ax[1].plot(fpr[i], tpr[i], label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")
        ax[1].plot(fpr["micro"], tpr["micro"], linestyle='--', color='black', label=f"Micro-average (AUC = {roc_auc['micro']:.2f})")

    ax[1].plot([0, 1], [0, 1], 'k--', lw=1)
    ax[1].set_title("ROC Curve")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].legend(loc="lower right")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

    plt.show()



