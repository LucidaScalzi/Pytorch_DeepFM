"""
Preprocess Criteo dataset. This dataset was used for the Display Advertising
Challenge (https://www.kaggle.com/c/criteo-display-ad-challenge).
"""
import os
import sys
import click
import random
import collections

# There are 13 integer features and 26 categorical features
continous_features = range(1, 14)
categorial_features = range(14, 40)

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
# 就是一个阈值的意思
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]

# 处理离散变量
class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            # 输出一个有 num_feature 个 defaultdict(int, {}) 的list
            self.dicts.append(collections.defaultdict(int))

    # 生成关于特征的dict
    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t') # Python rstrip() 删除 string 字符串末尾的指定字符（默认为空格）.
                for i in range(0, self.num_feature):
                    # features[categorial_features[i]] 指的是feature那些离散的变量中的第i个
                    if features[categorial_features[i]] != '':
                        # 相当于生成了一个关于 每个特征 每个特征中的某特征的取值个数的dict
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。前面是函数（注意是filter掉不满足的值），后面是序列
            # 在建立索引字典的时候我们会将 词频太低 的数据过滤，词频通过cutoff设置
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    # 我们的raw data数据是有存在一些缺少值的，我们对缺失值采取的手段是填0处理
    # 注意不只是缺失值会被填为0，一些词频较低的也会填充为0
    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]


# 处理连续变量
class ContinuousFeatureGenerator:
    """
    Clip continuous features.
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        # 离群点处理（过大值）
                        if val > continous_clip[i]:
                            val = continous_clip[i]

    # 我们的raw data数据是有存在一些缺少值的，我们对缺失值采取的手段是填0处理
    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return val


# @click.command("preprocess")
# @click.option("--datadir", type=str, help="Path to raw criteo dataset")
# @click.option("--outdir", type=str, help="Path to save the processed data")
def preprocess(datadir, outdir):
    """
    All the 13 integer features are normalzied to continous values and these
    continous features are combined into one vecotr with dimension 13.
    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """
    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(os.path.join(datadir, 'train.txt'), continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(
        os.path.join(datadir, 'train.txt'), categorial_features, cutoff=10)

    dict_sizes = dicts.dicts_sizes()
    # 改动的地方
    categorial_feature_offset = [0]
    for i in range(1, len(categorial_features)):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)

    # feature_sizes.txt
    with open(os.path.join(outdir, 'feature_sizes.txt'), 'w') as feature_sizes:
        # list和int不能运算，list + list即为两个list的合并
        # 连续变量就都是1， 离散变量就是其特征域大小
        sizes = [1] * len(continous_features) + dict_sizes
        sizes = [str(i) for i in sizes]
        feature_sizes.write(','.join(sizes))

    random.seed(0)

    # Saving the data used for training.
    with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        with open(os.path.join(datadir, 'train.txt'), 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                continous_vals = []
                for i in range(0, len(continous_features)):
                    # gen是填充空值
                    val = dists.gen(i, features[continous_features[i]])
                    continous_vals.append("{0:.6f}".format(val).rstrip('0')
                                            .rstrip('.'))
                categorial_vals = []
                for i in range(0, len(categorial_features)):
#                    val = dicts.gen(i, features[categorial_features[
#                        i]]) + categorial_feature_offset[i]
                    val = dicts.gen(i, features[categorial_features[i]]) #修改过
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                label = features[0]
                out_train.write(','.join([continous_vals, categorial_vals, label]) + '\n')
                    

    with open(os.path.join(outdir, 'test.txt'), 'w') as out:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    continous_vals.append("{0:.6f}".format(val).rstrip('0')
                                          .rstrip('.'))
                categorial_vals = []
                for i in range(0, len(categorial_features)):
#                    val = dicts.gen(i, features[categorial_features[
#                        i] - 1]) + categorial_feature_offset[i]
                    val = dicts.gen(i, features[categorial_features[i] - 1]) #修改过
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                out.write(','.join([continous_vals, categorial_vals]) + '\n')


# 运行下来就会在".\data"路径生成处理好的训练数据"train.txt"、测试数据"test.txt"以及特征表"feature_size.txt"

if __name__ == "__main__":
    # 这里要改一下，不是绝对路径就跑不出来
    preprocess('E:/github/Pytorch_DeepFM/data/raw/', 'E:/github/Pytorch_DeepFM/data/')
    # preprocess('../data/raw/', '../data')
    
#for test 0923

#datadir = '../data/raw'
#outdir = '../data'
#dicts = CategoryDictGenerator(len(categorial_features))
#dicts.build(
#    os.path.join(datadir, 'train.txt'), categorial_features, cutoff=10)
#dict_sizes,dict_test = dicts.dicts_sizes()






# 我们在使用embedding layer的时候切记三步走
# 一是建立索引字典
# 二是根据索引字典映射原始数据
# 三是根据索引字典得到feature_size之后才建立embedding layer


