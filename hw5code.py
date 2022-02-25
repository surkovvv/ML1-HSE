import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
    $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    # Заметки : пороги меньше min(feature) и больше max(feature) не рассматриваются
    pairs = np.array(sorted(zip(feature_vector, target_vector)))
    feature_vector = pairs[:, 0]
    target_vector = pairs[:, 1]
    k = target_vector.shape[0]

    feature_vector, index = np.unique(feature_vector, return_index=True)
    n = feature_vector.shape[0]
    index = index[1:n] - 1
    pref = np.cumsum(target_vector)

    thresholds = (feature_vector[: n - 1] + feature_vector[1 : n + 1]) / 2
    positive = pref[k - 1]
    R_l = index + 1
    R_r = k - R_l
    c1_l = pref[index]
    c1_r = positive - c1_l
    c0_l = R_l - c1_l
    c0_r = R_r - c1_r
    ginis = (((c0_l * c0_l + c1_l * c1_l) / R_l + (c0_r * c0_r + c1_r * c1_r) / R_r) / k) - 1

    best = np.argmax(ginis)

    return thresholds, ginis, thresholds[best], ginis[best]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]):
            node["height"] = 0
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if ((self._max_depth is not None) and self._max_depth <= depth) or \
                ((self._cur_min_samples_split is not None) and self._cur_min_samples_split >= sub_y.shape[0]):
            node["height"] = 0
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            ratio = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                feature_vector = np.array(list(map(lambda x: ratio[x], sub_X[:, feature])))
            else:
                raise ValueError

            if np.min(feature_vector) == np.max(feature_vector):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, ratio.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["height"] = 0
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        left_X = sub_X[split]
        left_y = sub_y[split]
        right_X = sub_X[np.logical_not(split)]
        right_y = sub_y[np.logical_not(split)]
        if (self._cur_min_samples_leaf is not None) and \
                left_y.shape[0] < self._cur_min_samples_leaf and \
                right_y.shape[0] < self._cur_min_samples_leaf:
            node["height"] = 0
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = set(threshold_best)
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(left_X, left_y, node["left_child"], depth + 1)
        self._fit_node(right_X, right_y, node["right_child"], depth + 1)
        node["height"] = max(node["left_child"]["height"], node["right_child"]["height"]) + 1

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def get_params(self, deep):
        return {'feature_types': self._feature_types,
                'max_depth': self._max_depth,
                'min_samples_split': self._min_samples_split,
                'min_samples_leaf': self._min_samples_leaf}

    def fit(self, X, y):
        self._cur_min_samples_split = self._min_samples_split
        self._cur_min_samples_leaf = self._min_samples_leaf
        if isinstance(self._min_samples_split, float):
            self._cur_min_samples_split = np.ceil(self._min_samples_split * y.shape[0])
        if isinstance(self._min_samples_leaf, float):
            self._cur_min_samples_leaf = np.ceil(self._min_samples_leaf * y.shape[0])
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_depth(self):
        return self._tree["height"]