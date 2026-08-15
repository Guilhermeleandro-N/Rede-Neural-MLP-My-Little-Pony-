import numpy as np


class Accuracy:

    @staticmethod
    def calculate(
        y_true,
        y_pred
    ):

        predictions = (
            y_pred > 0.5
        ).astype(int)

        return np.mean(
            predictions == y_true
        )

class Precision:

    @staticmethod
    def calculate(
        y_true,
        y_pred
    ):

        predictions = (
            y_pred > 0.5
        ).astype(int)

        true_positive = np.sum(
            (predictions == 1) &
            (y_true == 1)
        )

        false_positive = np.sum(
            (predictions == 1) &
            (y_true == 0)
        )

        if true_positive + false_positive == 0:
            return 0.0

        return (
            true_positive /
            (true_positive + false_positive)
        )


class Recall:

    @staticmethod
    def calculate(
        y_true,
        y_pred
    ):

        predictions = (
            y_pred > 0.5
        ).astype(int)

        true_positive = np.sum(
            (predictions == 1) &
            (y_true == 1)
        )

        false_negative = np.sum(
            (predictions == 0) &
            (y_true == 1)
        )

        if true_positive + false_negative == 0:
            return 0.0

        return (
            true_positive /
            (true_positive + false_negative)
        )


class F1Score:

    @staticmethod
    def calculate(
        y_true,
        y_pred
    ):

        precision = Precision.calculate(
            y_true,
            y_pred
        )

        recall = Recall.calculate(
            y_true,
            y_pred
        )

        if precision + recall == 0:
            return 0.0

        return (
            2 * precision * recall /
            (precision + recall)
        )