"""Different methods of clustering analysis."""
from skrules import SkopeRules
import sys
from typing import List, Union

import pandas as pd
import six
from IPython.display import HTML, display
from sklearn.tree import DecisionTreeClassifier, _tree

# workaround because skrules uses deprecated import from sklearn
sys.modules['sklearn.externals.six'] = six

# Code from https://towardsdatascience.com/the-easiest-way-to-interpret-clustering-result-8137e488a127


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    def tree_dfs(node_id=0, current_rule=[]):
        # feature[i] holds the feature to split on, for the internal node i.
        split_feature = inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED:  # internal node
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            tree_dfs(inner_tree.children_left[node_id], left_rule)
            # right child
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            tree_dfs(inner_tree.children_right[node_id], right_rule)
        else:  # leaf
            dist = inner_tree.value[node_id][0]
            dist = dist/dist.sum()
            max_idx = dist.argmax()
            if len(current_rule) == 0:
                rule_string = "ALL"
            else:
                rule_string = " and ".join(current_rule)
            # register new rule to dictionary
            selected_class = classes[max_idx]
            class_probability = dist[max_idx]
            class_rules = class_rules_dict.get(selected_class, [])
            class_rules.append((rule_string, class_probability))
            class_rules_dict[selected_class] = class_rules

    tree_dfs()  # start from root, node_id = 0
    return class_rules_dict


def cluster_report_DT(data: pd.DataFrame, clusters, min_samples_leaf=50, pruning_level=0.01):
    # Create Model
    tree = DecisionTreeClassifier(
        min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level)
    tree.fit(data, clusters)

    # Generate Report
    feature_names = data.columns
    class_rule_dict = get_class_rules(tree, feature_names)

    report_class_list = []
    for class_name in class_rule_dict.keys():
        rule_list = class_rule_dict[class_name]
        combined_string = ""
        for rule in rule_list:
            combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
        report_class_list.append((class_name, combined_string))

    cluster_instance_df = pd.Series(clusters).value_counts().reset_index()
    cluster_instance_df.columns = ['class_name', 'instance_count']
    report_df = pd.DataFrame(report_class_list, columns=[
                             'class_name', 'rule_list'])
    report_df = pd.merge(cluster_instance_df, report_df,
                         on='class_name', how='left')
    pretty_print(report_df.sort_values(by='class_name')[
                 ['class_name', 'instance_count', 'rule_list']])
# end of code from https://towardsdatascience.com/the-easiest-way-to-interpret-clustering-result-8137e488a127


def cluster_report_skrules(data: pd.DataFrame, clusters: Union[pd.Series, list]):
    for cluster in sorted(clusters.unique().tolist()):
        y_train = (clusters == int(cluster)) * 1

        skope_rules_clf = SkopeRules(
            feature_names=data.columns.tolist(),
            random_state=42,
            n_estimators=5,
            recall_min=0.5,
            precision_min=0.5,
            max_depth_duplication=0,
            max_samples=1.,
            max_depth=3
        )

        skope_rules_clf.fit(data, y_train)
        print(f'Cluster {str(cluster)}:')
        print(skope_rules_clf.rules_)
