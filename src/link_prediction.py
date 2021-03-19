import torch 
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, \
    average_precision_score, roc_auc_score

def fmt_cm(cm):
    spacing = [
        len(str(cm[i][0]))
        for i in range(2)
    ]

    cell_size = max(spacing)
    pad1 = cell_size-spacing[0]
    pad2 = cell_size-spacing[1]
    
    print("   P0%s  P1" % (' '* (cell_size-1)))
    print("T0 %d%s | %d" % (cm[0][0], ' '*pad1, cm[0][1]))
    print("T1 %d%s | %d" % (cm[1][0], ' '*pad2, cm[1][1]))


class LP_Classifier():
    def __init__(self, dumb_range=[0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.55]):
        self.model = SGDClassifier(loss='log', n_jobs=16)
        self.e = 0
        self.dumb_range=dumb_range

    def __call__(self, X):
        return self.model.predict(X)

    '''
    Uses SGD to train LR module on training data to predict if unknown
    edges are anomalous or not
    '''
    def train_lp_step(self, X_tr, y_tr, X_va, y_va):
        self.model.partial_fit(X_tr, y_tr, classes=[0,1])

        y_hat = self.model.predict(X_va)
        print('[%d]' % self.e)
        cm = confusion_matrix(y_va, y_hat)
        fmt_cm(cm)
        print()

        # TODO some sort of early stopping criterion 
        self.e += 1
        return cm 

    def score(self, y, y_hat):
        cr = classification_report(y, y_hat)
        cm = confusion_matrix(y, y_hat)

        print(cr)
        fmt_cm(cm)
        print()

    # Uses pure likelihood scores to predict
    def dumb_predict(self, probs, y):
        for i in range(100, 5, -1):
            i = float(i)/100
            cl = torch.zeros(probs.size(0))
            cl[probs > i] = 1

            print("%0.2f: " % i)
            cm = confusion_matrix(y, cl)
            fmt_cm(cm)

            if cm[0][0] == 0 and cm[1][0] == 0:
                break