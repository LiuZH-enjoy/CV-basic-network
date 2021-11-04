import jieba
import numpy as np
from collections import OrderedDict
import json


class Tokenizer:
    def __init__(self, ignore=["\n", "\t", "\r", " "], token_ids=["[null]", "[start]", "[end]", "[unk]"]):
        self.ignore = ignore
        self.token_ids = token_ids
        self.word_dict = OrderedDict()
        self.word_index = {token_ids[i]: i for i in range(len(token_ids))}
        self.max_len = None

    @classmethod
    def from_word_json(cls, file):
        token = Tokenizer()
        with open(file, "r", encoding="utf-8") as f:
            token.word_index = json.load(f)
        token.max_len = int(token.word_index["max_len"])
        return token

    def jieba_word(self, text):
        return " ".join(list(jieba.cut(text)))

    def add_word_index(self, word):
        if word not in self.ignore:
            if word not in self.word_dict:
                self.word_dict[word] = 1
            else:
                self.word_dict[word] += 1

    def add_word_num(self, word, encoder):
        try:
            encoder.append(self.word_index[word])
        except:
            encoder.append(self.word_index["[unk]"])
        return encoder


    def fit_text(self, text, is_jieba=False, space_split=False, save_file=False):

        if is_jieba:
            space_split = True
            text = [self.jieba_word(i) for i in text]

        if space_split:
            self.max_len = max([len(i.split(" ")) for i in text]) + 4
        else:
            self.max_len = max([len(i) for i in text]) + 4

        for sentence in text:
            if not space_split:
                for word in sentence:
                    self.add_word_index(word)
            else:
                for word in sentence.split(" "):
                    self.add_word_index(word)

        word_index = list(self.word_dict.items())
        word_index.sort(key=lambda x:x[1], reverse=True)

        for k, _ in word_index:
            self.word_index[k] = len(self.word_index)
        self.word_index["max_len"] = str(self.max_len)
        if save_file:

            with open("word_index.json", "w", encoding="utf-8") as f:
                json.dump(self.word_index, f, ensure_ascii=False, indent=4)

    def encoder_sentence(self, sentence, space_split=False, add_token_id=True):

        result = []
        for text in sentence:
            encoder = []
            if not space_split:
                for word in text:
                    encoder = self.add_word_num(word, encoder)
            else:
                for word in text.split(" "):
                    encoder = self.add_word_num(word, encoder)
            encoder.append(self.word_index["[end]"])
            encoder.insert(0, self.word_index["[start]"])
            result.append(encoder)
        return result

    def padding(self, sentence, pad_num=0, padding_kind="post", max_len=-1):
        if max_len == -1:
            max_len = self.max_len

        mask = np.zeros((len(sentence), max_len), dtype="int32")

        if padding_kind == "post":
            for i, nums in enumerate(sentence):
                nums_len = len(nums)
                if nums_len > max_len:
                    mask[i, :] = nums[:max_len]
                else:
                    mask[i, :nums_len] = nums
        else:
            for i, nums in enumerate(sentence):
                nums_len = len(nums)
                if nums_len > max_len:
                    mask[i, :] = nums[:max_len]
                else:
                    mask[i, -nums_len:] = nums
        return mask

    def decoder_nums(self, arr, save_token=False):
        arr_shape = arr.shape
        sentences = []
        index_word = {v:k for k, v in self.word_index.items()}
        if len(arr_shape) == 2:
            for sentence_arr in arr:
                sentence = []
                for num in sentence_arr:
                    word = index_word[num]
                    if not save_token:
                        if word in self.token_ids:
                            continue
                        sentence.append(word)
                    else:
                        sentence.append(word)

                sentences.append("".join(sentence))
        elif len(arr_shape) == 1:
            sentence = []

            for num in arr:
                word = index_word[num]
                if not save_token:
                    if word in self.token_ids:
                        continue
                    sentence.append(word)
                else:
                    sentence.append(word)

            sentences.append("".join(sentence))
        else:
            assert "维度超过二了，最大的维度只能是二维"
        return sentences

if __name__ == '__main__':
    token = Tokenizer()
    sentences = ["我爱深度学习", "我爱自然语言处理", "我爱计算机视觉处理"]
    token.fit_text(sentences, save_file=True)
    data = token.encoder_sentence(sentences)
    data = token.padding(data, padding_kind="padding")
    sentence = token.decoder_nums(data)





