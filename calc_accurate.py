from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_recall_fscore_support
import numpy as np


class Evaluator:
    def __init__(self, dev_y):
        self.test_y_org = dev_y

    def calc_accuracy(self, dev_pred, test_pred, factor_num):
        # 四捨五入
        dev_pred_int = np.rint(dev_pred).astype('int32')
        test_pred_int = np.rint(test_pred).astype('int32')
        # print(self.test_y_org[factor_num], dev_pred_int)
        dev_accuracy = accuracy_score(
            self.test_y_org[factor_num], dev_pred_int)
        test_accuracy = accuracy_score(
            self.test_y_org[factor_num], test_pred_int)
        return dev_accuracy, test_accuracy
        # precision_recall_fscore_support(y_true, y_pred, average='micro')


def test_calc_accuracy():
    dev_y = np.arange(20).reshape(2, -1)
    dev_query = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1)
    test_query = np.array([0, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1)
    gold = np.array([0, 11, 21, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                     11, 11, 11, 11, 11, 11, 11, 11, 11, 11]).reshape(2, -1)
    dev_target = 9/11
    test_target = 10/11
    factor_num = 0
    eval = Evaluator(gold)
    dev_acc, test_acc = eval.calc_accuracy(dev_query, test_query, factor_num)
    assert dev_acc == dev_target and test_acc == test_target
    print(dev_acc, test_acc)


if __name__ == "__main__":
    test_calc_accuracy()
