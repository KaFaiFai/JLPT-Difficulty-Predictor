import numpy as np
import sklearn.metrics

class ClassificationMetrics:
    def __init__(self, truths, outputs):
        """
        truths: (N), outputs: (N, C)
        """

        # process inputs
        truths_arr = np.array(truths).squeeze()
        outputs_arr = np.array(outputs).squeeze()

        self.truths = truths_arr  # ground truth predictions
        self._outputs = outputs_arr
        self.preds = np.argmax(outputs_arr, axis=1)

    def print_report(self, topK=None):
        accuracy = sklearn.metrics.accuracy_score(self.truths, self.preds)
        precision = sklearn.metrics.precision_score(self.truths, self.preds, average="macro", zero_division=0)
        recall = sklearn.metrics.recall_score(self.truths, self.preds, average="macro", zero_division=0)
        f1_score = sklearn.metrics.f1_score(self.truths, self.preds, average="macro", zero_division=0)

        if topK is None:
            print(f"Accuracy: {accuracy:2.2%} | Precision: {precision:.4f}")
            print(f"Recall:   {recall:.4f} | F1 score:  {f1_score:.4f}")
        else:
            topK_accuracy = sklearn.metrics.top_k_accuracy_score(self.truths, self._outputs, k=topK)
            print(f"Top 1 Accuracy: {accuracy:2.2%} | Top 5 Accuracy: {topK_accuracy:2.2%}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 score: {f1_score:.4f}")
