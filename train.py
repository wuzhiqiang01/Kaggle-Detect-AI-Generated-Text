import gc
import pandas as pd
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFC, Lowercase
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# 参考
# https://www.kaggle.com/code/datafan07/train-your-own-tokenizer/notebook
class CFG:
    test_path = "datasets/llm-detect-ai-generated-text/test_essays.csv"
    train_path = "datasets/llm-detect-ai-generated-text/train_essays.csv"
    sub_path = "datasets/llm-detect-ai-generated-text/sample_submission.csv"
    extra_data_path = "datasets/llm-detect-ai-generated-text/train_v2_drcat_02.csv"
    additional_path = None
    LOWER_CASE = False
    VOCAB_SIZE = 30522 


def train_tokenizers(train_df, test_df):

    # tokenizer处理的过程可以参考
    # https://huggingface.co/learn/nlp-course/chapter6/4?fw=pt
    # Normalization Pre-tokeniation Model PostProcesser

    # tokenizers库的参考地址
    # https://huggingface.co/docs/tokenizers/quicktour
    raw_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    raw_tokenizer.normalizer = Sequence([NFC()] + [Lowercase()] if CFG.LOWER_CASE else [])
    raw_tokenizer.pre_tokenizer = ByteLevel()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = BpeTrainer(vocab_size=CFG.VOCAB_SIZE, special_tokens=special_tokens)  # 构建训练的BPE

    # test_df['text']与test_df[['text']]的不同
    # 前者数据类型为<class 'pandas.core.series.Series'>
    # 后者数据类型为<class 'pandas.core.frame.DataFrame'>

    all_df = pd.concat([test_df[['text']], train_df[['text']]])    
    dataset = Dataset.from_pandas(all_df)
    def train_corp_iter(): 
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    # 训练tokenizer
    raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    # 对与测试数据进行分词
    tokenized_texts_train = []
    for text in tqdm(train_df['text'].tolist()):
        tokenized_texts_train.append(tokenizer.tokenize(text))
    
    # 对训练数据进行分词
    tokenized_texts_test = []
    for text in tqdm(test_df['text'].tolist()):
        tokenized_texts_test.append(tokenizer.tokenize(text))

    return tokenized_texts_train, tokenized_texts_test


def tf_idf(tokenized_texts_train, tokenized_texts_test):
    def dummy(text):
        return text
    
    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        lowercase=False, sublinear_tf=True, analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None, strip_accents='unicode')

    vectorizer.fit(tokenized_texts_test)

    # Getting vocab
    vocab = vectorizer.vocabulary_

    # print(vocab)

    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                                analyzer = 'word',
                                tokenizer = dummy,
                                preprocessor = dummy,
                                token_pattern = None, strip_accents='unicode'
                                )

    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    tf_test = vectorizer.transform(tokenized_texts_test)

    del vectorizer
    gc.collect()
    return tf_train, tf_test


def main():
    df_train = pd.read_csv(CFG.train_path)
    df_test  = pd.read_csv(CFG.test_path)
    df_train.drop("generated", axis=1, inplace=True)
    # df_sub = pd.read_csv(CFG.sub_path)
    df_test = df_train
    df_sub = df_train
    df_extra_data = pd.read_csv(CFG.extra_data_path, sep=",")

    df_extra_data = df_extra_data.drop_duplicates(subset=['text'])
    df_extra_data.reset_index(drop=True, inplace=True)
    y_train = df_extra_data['label'].values
    tokenized_texts_train, tokenized_texts_test = train_tokenizers(df_extra_data, df_test)
    tf_train, tf_test = tf_idf(tokenized_texts_train, tokenized_texts_test)

    if len(df_test.text.values) <= 5:
        df_sub.to_csv('submission.csv', index=False)
    else:
        clf = MultinomialNB(alpha=0.1)
        sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber") 
        p6={'n_iter': 2500,
            'verbose': -1,
            'objective': 'cross_entropy',
            'metric': 'auc',
            'learning_rate': 0.01, 
            'colsample_bytree': 0.78,
            'colsample_bynode': 0.8, 
            'lambda_l1': 4.562963348932286, 
            'lambda_l2': 2.97485, 
            'min_data_in_leaf': 115, 
            'max_depth': 23, 
            'max_bin': 898}
        
        lgb=LGBMClassifier(**p6)
        cat=CatBoostClassifier(
            iterations=2000,
            verbose=0,
            l2_leaf_reg=6.6591278779517808,
            learning_rate=0.1,
            subsample = 0.4,
            allow_const_label=True,loss_function = 'CrossEntropy')
        weights = [0.068,0.311,0.31,0.311]
    
        ensemble = VotingClassifier(estimators=[('mnb',clf),
                                                ('sgd', sgd_model),
                                                ('lgb',lgb), 
                                                ('cat', cat)
                                            ],
                                    weights=weights, voting='soft', n_jobs=-1)
        ensemble.fit(tf_train, y_train)
        gc.collect()
        final_preds = ensemble.predict_proba(tf_test)[:,1]
        df_sub['generated'] = final_preds
        df_sub.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()