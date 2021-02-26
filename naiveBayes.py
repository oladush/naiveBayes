import  pandas as pd

df = pd.read_csv("mlbootcamp5_train.csv",
                 sep=";",
                 index_col="id")

class NaiveBayes:
    def __init__(self):
        self.BayesTheorem = lambda pcx, px, pc: pcx * pc / px

    def train(self, dataset, train, target, target_val):
        self.dataset = dataset
        self.target = target
        self.train = train

        pc = dataset[dataset[target] == target_val].shape[0] / dataset.shape[0]

        pcx = {}
        px = {}
        for tag in train:
            pcx[tag] = {}
            px[tag] = {}
            for val in dataset[tag].unique():
                pcx[tag][val] = self.findPCX(
                    target, tag, target_val, val
                )
                px[tag][val] = self.findPX(
                    tag, val
                )

        self.pc = pc
        self.pcx_cost = pcx
        self.px_cost = px

        print(pcx)
        print(px)
        return pcx, pc, px

    def classify(self, newset):
        dataset = newset[self.train]
        print(dataset)
        ver = [0] * (dataset.shape[0] + 1)

        for index, row in dataset.iterrows():
            px = 1
            pcx = 1
            for tag in self.train:
                value = row[tag]
                pcx *= self.pcx_cost[tag][value]
                px *= self.px_cost[tag][value]

            # print(row['cholesterol'])
            try:
                ver[index] = self.BayesTheorem(pcx, px, self.pc)
            except:
                pass

        return ver


    def findPCX(self, class_c, class_x, event_c, event_x):
        print(class_c)
        dst = self.dataset[[class_x, class_c]]

        al = dst[dst[class_c] == event_c]
        co = al[al[class_x] == event_x]

        return co.shape[0] / al.shape[0]


    def findPX(self, class_x, event_x):
        xo = self.dataset[self.dataset[class_x] == event_x].shape[0]
        al = self.dataset.shape[0]

        return xo / al
