import pandas as pd
from datasets import Dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


class CFG:
    test_path = "datasets/llm-detect-ai-generated-text/test_essays.csv"
    train_path = "datasets/llm-detect-ai-generated-text/train_essays.csv"
    additional_path = None
    LOWER_CASE = False
    VOCAB_SIZE = 30522 


def train_tokenizers(train_df, test_df):

    # token处理的过程可以参考https://huggingface.co/learn/nlp-course/chapter6/4?fw=pt
    # Normalization Pre-tokeniation Model PostProcesser
    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if CFG.LOWER_CASE else [])
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=CFG.VOCAB_SIZE, special_tokens=special_tokens)  # 构建训练的BPE
    dataset = Dataset.from_pandas(test_df[['text']])
    def train_corp_iter(): 
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]
    raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenized_texts_test = []

    for text in tqdm(test_df['text'].tolist()):
        tokenized_texts_test.append(tokenizer.tokenize(text))

    tokenized_texts_train = []

    for text in tqdm(train_df['text'].tolist()):
        tokenized_texts_train.append(tokenizer.tokenize(text))
    print(tokenized_texts_test)

def main():
    df_train = pd.read_csv(CFG.train_path)
    df_test  = pd.read_csv(CFG.test_path)

    df_train = df_train.drop_duplicates(subset=['text'])
    df_train.reset_index(drop=True, inplace=True)

    train_tokenizers(df_train, df_test)


if __name__ == "__main__":
    main()