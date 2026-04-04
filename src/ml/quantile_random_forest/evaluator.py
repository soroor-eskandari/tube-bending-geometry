import numpy as np


class QRFEvaluator:

    @staticmethod
    def coverage(y_true, lower, upper):
        return np.mean((y_true >= lower) & (y_true <= upper))

    @staticmethod
    def width(lower, upper):
        return np.mean(upper - lower)

    @staticmethod
    def evaluate(y_true, preds):
        return {
            "coverage": QRFEvaluator.coverage(
                y_true, preds["lower"], preds["upper"]
            ),
            "avg_width": QRFEvaluator.width(
                preds["lower"], preds["upper"]
            )
        }