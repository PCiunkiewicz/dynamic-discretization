
import numpy as np
import pandas as pd
import pyAgrum as gum

from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import roc_auc_score
from pyAgrum.lib.bn2roc import __computepoints as computepoints


class CustomBayesNet:
    """
    Custom wrapper class for PyAgrum for parameter-learning
    Bayesian networks.
    """
    def __init__(self, data, train_file='train.csv', test_file='test.csv'):
        """
        Params
        ------
        data: pandas.DataFrame
            Dataset you are working with.

        train_file: string (optional)
            Path to training data for PyAgrum.

        test_file: string (optional)
            Path to testing data for PyAgrum.
        """
        self.bn = gum.BayesNet("CustomBayesNet")
        self.train = train_file
        self.test = test_file

        self.data = data.copy()
        for col in self.data:
            self.bn.add(gum.LabelizedVariable(col, col, self._sort(self.data[col].unique())))

    def _remove_unused_nodes(self):
        """
        Convenience method for removing unused nodes from
        the Bayesian network DAG.
        """
        used_nodes = self.bn.connectedComponents()[0]
        all_nodes = self.bn.nodes()
        unused = all_nodes.difference(used_nodes)

        for node in unused:
            self.bn.erase(node)

        for file in [self.train, self.test]:
            data = pd.read_csv(file)[self.bn.names()]
            data.to_csv(file, index=False)

        self.data = self.data[self.bn.names()]
        self.bn = gum.BayesNet("CustomBayesNet")
        for col in self.data:
            self.bn.add(gum.LabelizedVariable(col, col, self._sort(self.data[col].unique())))
        self.topology()

    def _sort(self, vals):
        """
        Convenience method for sorting numerical intervals
        for future representation in the NPT graph.

        Params
        ------
        vals: array_like
            Array of generated string labels for sorting.
            DynamicDiscretizer and StatisDiscretizer classes
            are recommended for generating these labels.

        Returns
        -------
        vals: array_like
            Sorted string labels.
        """
        if all([';' in x for x in vals]):
            num = [float(x.split(';')[0][1:]) for x in vals]
            order = np.argsort(num)
            return [vals[i] for i in order]
        else:
            return sorted(vals)

    def report_score(self, target, label):
        """
        BN scoring method. Computes various binary classification
        metrics including precision, recall, specificity, F1
        score, geometric score, index of balanced accuracy,
        and ROC AUC.

        Params
        ------
        target: string
            Target node for scoring predictions. Must be binary.

        label: string
            Target positive label. Must exist in `target` node.

        Returns
        -------
        results: dict
            Dictionary of score values.
        """
        pred = computepoints(self.bn, self.test, target, label)[0]
        y_test = np.array(pred)[:,1] == label
        y_proba = np.array(pred)[:,0].astype(float)
        y_pred = y_proba > 0.5

        res = classification_report_imbalanced(y_test, y_pred, output_dict=True)
        res['avg_roc_auc'] = roc_auc_score(y_test, y_proba)

        return {k[4:]:v for k,v in res.items() if 'avg_' in str(k)}

    def construct(self, train, test):
        """
        Bayesian network construction and parameter learning.

        Params
        ------
        train: pandas.DataFrame
            Training data for saving to file.

        test: pandas.DataFrame
            Testing data for saving to file.

        Returns
        -------
        self: CustomBayesNet
            CustomBayesNet instance with Bayesian network
            constructed and learned from data.
        """
        train.to_csv(self.train, index=False)
        test.to_csv(self.test, index=False)

        self.topology()
        self._remove_unused_nodes()
        learner = gum.BNLearner(self.train, self.bn)
        self.bn = learner.learnParameters(self.bn.dag())

        return self

    def topology(self):
        """
        Network topology defined when subclassing CustomBayesNet.
        """
        pass


class ToxicityBN(CustomBayesNet):
    """
    Custom Bayesian network for predicting worsening normal
    tissue toxicity from the ACCEL clinical trial.
    """
    def __init__(self, data, train_file='train.csv', test_file='test.csv'):
        """
        Params
        ------
        data: pandas.DataFrame
            Dataset you are working with.

        train_file: string (optional)
            Path to training data for PyAgrum.

        test_file: string (optional)
            Path to testing data for PyAgrum.
        """
        super().__init__(data, train_file, test_file)

    def topology(self):
        """
        Network topology defined when subclassing CustomBayesNet.
        """
        for var in ['Age', 'Smokes', 'Vol']:
            self.bn.addArc(var, 'Size')

        self.bn.addArc('Size', 'GTV')
        self.bn.addArc('GTV', 'CTV')
        self.bn.addArc('CTV', 'PTV')

        for var in ['Size', 'DCIS', 'Lymph']:
            self.bn.addArc(var, 'Grade')

        for var in ['Grade', 'PTV']:
            self.bn.addArc(var, 'DEV_PTV')

        for var in ['DEV_PTV', 'Cos_0', 'Fib_0']:
            self.bn.addArc(var, 'Toxicity')

        for var in ['Cos_6wk']:
            self.bn.addArc(var, var[:4] + '1yr')
            self.bn.addArc('Toxicity', var)


class BreastCancerBN(CustomBayesNet):
    """
    Custom Bayesian network for predicting malignant
    vs benign tumors for Wisconsis breast cancer data.
    """
    def __init__(self, data, train_file='train_cancer.csv', test_file='test_cancer.csv'):
        """
        Params
        ------
        data: pandas.DataFrame
            Dataset you are working with.

        train_file: string (optional)
            Path to training data for PyAgrum.

        test_file: string (optional)
            Path to testing data for PyAgrum.
        """
        super().__init__(data, train_file, test_file)

    def topology(self):
        """
        Network topology defined when subclassing CustomBayesNet.
        """
        for quantity in ['symmetry', 'fractal dimension']:
            self.bn.addArc(f'worst {quantity}', f'mean {quantity}')
            self.bn.addArc(f'mean {quantity}', 'malignant')

        self.bn.addArc('worst perimeter', 'mean perimeter')
        self.bn.addArc('mean perimeter', 'worst radius')
        self.bn.addArc('worst radius', 'mean radius')
        self.bn.addArc('mean radius', 'worst area')
        self.bn.addArc('worst area', 'mean area')

        self.bn.addArc('worst concavity', 'mean concavity')
        self.bn.addArc('mean concavity', 'worst concave points')
        self.bn.addArc('worst concave points', 'mean concave points')

        self.bn.addArc('worst smoothness', 'mean smoothness')
        self.bn.addArc('mean smoothness', 'worst texture')
        self.bn.addArc('worst texture', 'mean texture')
        self.bn.addArc('mean texture', 'malignant')

        self.bn.addArc('mean area', 'worst compactness')
        self.bn.addArc('mean concave points', 'worst compactness')
        self.bn.addArc('worst compactness', 'mean compactness')
        self.bn.addArc('mean compactness', 'malignant')


class DiabetesBN(CustomBayesNet):
    """
    Custom Bayesian network for predicting high severity
    diabetes disease progression for diabetes data.
    """
    def __init__(self, data, train_file='train_diabetes.csv', test_file='test_diabetes.csv'):
        """
        Params
        ------
        data: pandas.DataFrame
            Dataset you are working with.

        train_file: string (optional)
            Path to training data for PyAgrum.

        test_file: string (optional)
            Path to testing data for PyAgrum.
        """
        super().__init__(data, train_file, test_file)

    def topology(self):
        """
        Network topology defined when subclassing CustomBayesNet.
        """
        self.bn.addArc('age', 'bmi')
        self.bn.addArc('sex', 'bmi')
        self.bn.addArc('bmi', 'bp')
        self.bn.addArc('s1', 'bp')
        self.bn.addArc('s4', 'bp')
        self.bn.addArc('bp', 'disease')

        for var in ['s2', 's3', 's5', 's6']:
            self.bn.addArc(var, 'disease')


class IrisBN(CustomBayesNet):
    """
    Custom Bayesian network for predicting species
    versicolor for iris data.
    """
    def __init__(self, data, train_file='train_iris.csv', test_file='test_iris.csv'):
        """
        Params
        ------
        data: pandas.DataFrame
            Dataset you are working with.

        train_file: string (optional)
            Path to training data for PyAgrum.

        test_file: string (optional)
            Path to testing data for PyAgrum.
        """
        super().__init__(data, train_file, test_file)

    def topology(self):
        """
        Network topology defined when subclassing CustomBayesNet.
        """
        for var in ['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)']:
            self.bn.addArc(var, 'versicolor')
