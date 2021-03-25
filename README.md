# Innovation-Competition
Solution for the TianChi competition 全球人工智能技术创新大赛【赛道三】
## Competition Address
https://tianchi.aliyun.com/competition/entrance/531851/introduction?spm=5176.12281949.1003.5.34bf24483IFs55
## Download Dataset
1.https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531851/20210228%E6%95%B0%E6%8D%AE%E6%9B%B4%E6%96%B0/gaiic_track3_round1_train_20210228.zip

2.https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531851/20210228%E6%95%B0%E6%8D%AE%E6%9B%B4%E6%96%B0/gaiic_track3_round1_testA_20210228.zip

## Preparation

### Environment 

if you have installed anaconda, open anaconda prompt to create a new python environment called **innovation** and install needed packages from privided **environment.yml** file in this repo.

```
conda env create -f environment.yml
```

You could also create the environment and install packages manually using the follow commands if you couldn't do it successfully.

**Note**:

if you don't have GPU with cuda , you could use 'conda install pytorch torchvision torchaudio cpuonly -c pytorch' instead.

if you conda channels unavailable for transformers, use 'pip install transformers' command instead.

```
conda create -n innovation python=3.8
conda activate innovation
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
conda install -c huggingface transformers
conda install notebook
conda install pandas
conda install matplotlib
conda install -c conda-forge scikit-learn 
pip install pytorch-lightning==1.2.0
pip install PyYAML
```

### Git

clone a remote repository at your local machine.

```
git clone https://github.com/naive-one/Innovation-Competition
```

if you are fresh for git, read the simple guide for git at the end. 

## Train

### dataset process

1.把BERT的Embedding层中的[UNK]、[CLS]、[SEP]等特殊标记拿出来，这部分不变，其余部分根据统计的密文数据和明文数据的字频进行初始化，可以加快加快模型的收敛速度。

2.数据增强，q1-q2=1,q1-q3=1 ---> q1-q3=1。q1-q2=1,q1-q3=0 ---> q1-q3=0

3.数据对偶，交换q1和q2位置

### mask

1.ngram mask

2.random mask

### model

1.不建议用哈工大开源的[RoBERTa-wwm-ext-large](https://github.com/ymcui/Chinese-BERT-wwm)，因为这个版本随机初始化了MLM部分的权重。可以用腾讯UER开源的[BERT Large](https://share.weiyun.com/5G90sMJ)。

2.华为NeZHa

### adversarial Training（对抗训练）

1.PGD

2.FGM

### pretrain task

1.训练数据做MLM和句子相似性任务（[cls]的输出作为预测），测试数据只做MLM任务，同时进行训练。

### fine-tune

1.k-fold

## Simple Guide for Git

your local repository consists of three "trees" maintained by git. the first one is your **Working Directory** which holds the actual files. the second one is the **Index** which acts as a staging area and finally the **HEAD** which points to the last commit you've made.

you could use 'add' command move file from working directory to index.

```
git add <filename>
```

you could use 'commit' command move file from index to head and take some words what you want to say.

```
git commit -m 'what are you changed'
```

Your changes are now in the HEAD of your local working copy. To send those changes to your remote repository.

```
git push origin main
```

create a new branch.

```
git checkout -b new_branch_name
```

switch back to main.

```
git checkout main
```

delete the branch.

```
git branch -d branch_name
```

to update your local repository to the newest commit, execute

```
git pull
```

to merge another branch into your active branch (e.g. main), use

```
git merge <branch>
```

you could also check https://git-scm.com/book/zh/v2 for more information about git.