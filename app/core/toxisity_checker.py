# import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

"""
#Dowload all models from Huggingface one time and cached:

from huggingface_hub import snapshot_download

path_to_rus_model = snapshot_download("cointegrated/rubert-tiny-toxicity")
path_to_eng_model = snapshot_download("citizenlab/distilbert-base-multilingual-cased-toxicity")

# path_to_rus_model now contains path to cached rus model
# path_to_eng_model now contains path to cached eng model

# need to save paths and use any time
# More about CACH system -  
# https://huggingface.co/docs/huggingface_hub/guides/manage-cache#manage-huggingfacehub-cachesystem
"""
# Dowload all models online from Huggingface everytime
path_to_rus_model = 'cointegrated/rubert-tiny-toxicity'
path_to_eng_model = 'citizenlab/distilbert-base-multilingual-cased-toxicity'


# Loading models takes time anyway (even from cach), so leave it outside of class object creation.
# Load ru model directly
model_ru = AutoModelForSequenceClassification.from_pretrained(path_to_rus_model)
tokenizer_ru = AutoTokenizer.from_pretrained(path_to_rus_model)

# Use a pipeline as a high-level helper for en model
text2toxicity_en = pipeline("text-classification", model=path_to_eng_model)


class SafeNaming:
    """This class contains functions to check if the typed sentence is toxic or just contains a personal name"""

    def __init__(self, toxity_level=0.5, silent=True):
        """Constructor"""

        # Dowload all models online from Huggingface everytime
        path_to_rus_model = 'cointegrated/rubert-tiny-toxicity'
        path_to_eng_model = 'citizenlab/distilbert-base-multilingual-cased-toxicity'
        model_ru = AutoModelForSequenceClassification.from_pretrained(path_to_rus_model)
        tokenizer_ru = AutoTokenizer.from_pretrained(path_to_rus_model)
        text2toxicity_en = pipeline("text-classification", model=path_to_eng_model)

        self.silent = silent

        # Load ru model directly
        self.tokenizer_ru = tokenizer_ru
        self.model_ru = model_ru
        if torch.cuda.is_available():
            self.model_ru.cuda()
        self.toxity_level = toxity_level

        # Use a pipeline as a high-level helper for en model
        self.text2toxicity_en = text2toxicity_en  # Get label of toxicity from english/"""

    def text2toxicity_ru(self, srting):
        """ Calculate  a vector of toxicity aspects on russian"""
        with torch.no_grad():
            inputs = self.tokenizer_ru(srting, return_tensors='pt', truncation=True, padding=True).to(
                self.model_ru.device)
            proba = torch.sigmoid(self.model_ru(**inputs).logits).cpu().numpy()
        return proba[0]

    def prepair(self, string):
        return string.lower().replace('-', ' ').strip()

    def transliteration_sumbol_ru_en(self, string):
        ru = "авекмнорстухАВЕКМНОРСТУХ"  # ru МН
        en = "abekmhopctuxABEKMHOPCTYX"  # en MH
        transliteration_ru_en = dict(zip(ru, en))

        # идем по пересечению ключей словаря и симолов в слове
        for i in set(string).intersection(set(transliteration_ru_en.keys())):
            string = string.replace(i, transliteration_ru_en[i])
        return string

    def transliteration_sumbol_en_ru(self, string):
        ru = "авекмнорстухАВЕКМНОРСТУХ"  # ru МН
        en = "abekmhopctuxABEKMHOPCTYX"  # en MH
        transliteration_en_ru = dict(zip(en, ru))

        # идем по пересечению ключей словаря и симолов в слове
        for i in set(string).intersection(set(transliteration_en_ru.keys())):
            string = string.replace(i, transliteration_en_ru[i])
        return string

    def transliteration_sound_en_ru(self, string):
        transliteration_en_ru = {'sch': 'щ', 'sh': 'ш', 'ch': 'ч', 'ya': 'я', 'yo': 'ё', 'b': 'б', 'v': 'в', 'w': 'в',
                                 'g': 'г', 'd': 'д', 'e': 'е',
                                 'zh': 'ж', 'k': 'к', 'l': 'л', 'm': 'м', 'p': 'п', 'r': 'р', 't': 'т', 'f': 'ф',
                                 'n': 'н',
                                 's': 'с',
                                 'c': 'с',
                                 'z': 'з',
                                 'a': 'а', 'o': 'о',
                                 'i': 'и', 'ja': 'я',
                                 'j': 'и',
                                 'u': 'у', 'y': 'ы', 'h': 'х', 'x': 'х', 'q': 'ку'}
        transliteration_ordered_keys = ['sch', 'sh', 'ch', 'ya', 'yo', 'ja', 'a', 'e', 'o', 'i', 'y', 'u', 'c', 's',
                                        'g', 'd', 'k', 'l', 'm', 'n', 'b', 'p', 'r', 't',
                                        'f', 'x', 'v', 'zh', 'h', 'z', 'j', 'q']

        # Здесь важен порядок перебора ключей словаря т.к. есть взаимовключающие наборы символов (etc. 'sch', 'sh', 's').
        # Важно использовать версию питона от 3.7, в которой сохраняется порядок перебора ключей ИЛИ вот такой список,
        # что выше и по этой же причине не идем по пересечению ключей словаря и симолов в слове

        for i in transliteration_ordered_keys:
            string = string.replace(i, transliteration_en_ru[i])
        return string

    def transliteration_sound_ru_en(self, string):
        transliteration_ru_en = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
                                 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'i', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
                                 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h',
                                 'ц': 'c', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e',
                                 'ю': 'u', 'я': 'ya', 'ґ': '', 'ї': 'i', 'є': 'e', 'Є': 'e'}

        # идем по пересечению ключей словаря и симолов в слове
        for i in set(string).intersection(set(transliteration_ru_en.keys())):
            string = string.replace(i, transliteration_ru_en[i])
        return string

    def check_russian(self, string):
        """ Check of toxicity on russian"""
        toxicity_evaluation = self.text2toxicity_ru(string)
        if not self.silent:
            print('ru: ', string)
        if any(i > self.toxity_level for i in toxicity_evaluation[1:]):
            return True
        else:
            return False

    def check_english(self, string):
        """ Check of toxicity on english/.."""
        if not self.silent:
            print('en: ', string)
        # if all('a'<=x<='z' for x in string):

        toxicity_evaluation = self.text2toxicity_en(string)[0]
        if toxicity_evaluation['label'] == 'toxic':
            return True
        else:
            return False

    def full_check_for_all_russian(self, string):
        if self.check_russian(string):  # проверка как русского слова
            return True

        if self.check_english(
                self.transliteration_sound_ru_en(string)):  # звуковая транслитерация и проверка как аглийского слова
            return True

        if self.check_english(
                self.transliteration_sumbol_ru_en(string)):  # символьная транслитерация и проверка как аглийского слова
            return True

        return False

    def full_check_for_all_english(self, string):
        if self.check_english(string):  # проверка как аглийского слова
            return True

        if self.check_russian(
                self.transliteration_sound_en_ru(string)):  # звуковая транслитерация и проверка как русского слова
            return True

        if self.check_russian(
                self.transliteration_sumbol_en_ru(string)):  # символьная транслитерация и проверка как русского слова
            return True

        return False

    def check_safety(self, string: str) -> str:

        if isinstance(string, str):
            string = self.prepair(string)

            if all('а' <= x <= 'я' or x == ' ' or x == '-' for x in string):  # проверка что все символы рус.
                if not self.silent:
                    print('Check all rus: ')
                return self.full_check_for_all_russian(string)

            elif all('a' <= x <= 'z' or x == ' ' or x == '-' for x in string):  # проверка что все символы анг.
                if not self.silent:
                    print('Check all eng: ')
                return self.full_check_for_all_english(string)

            else:  # разные символы
                if not self.silent:
                    print('Error. The input is mix languages')
                # return False
                if self.full_check_for_all_russian(self.transliteration_sumbol_en_ru(string)):
                    return True

                elif self.full_check_for_all_english(self.transliteration_sumbol_ru_en(string)):
                    return True

                else:
                    return False
        else:
            print('Error. The input is not a string')


class SafeNamingPromptQ(SafeNaming):
    """This class adds question 'Tебя зовут '+string+'?' when check toxity on russian"""

    def check_russian(self, string):
        """ Check of toxicity on russian"""
        toxicity_evaluation = self.text2toxicity_ru('Tебя зовут '+string+'?')
        if not self.silent:
            print('ru: ', string)
        if any(i > self.toxity_level for i in toxicity_evaluation[1:]):
            return True
        else:
            return False


class SafeNamingTitle(SafeNaming):
    """This class turn string to title when check toxity on russian"""

    def check_russian(self, string):
        """ Check of toxicity on russian"""
        toxicity_evaluation = self.text2toxicity_ru(string.title())
        if not self.silent:
            print('ru: ', string)
        if any(i > self.toxity_level for i in toxicity_evaluation[1:]):
            return True
        else:
            return False


class SafeNamingPromptQTitle(SafeNaming):
    """This class adds question 'Tебя зовут '+string+'?' and turn string to title when check toxity on russian"""

    def check_russian(self, string):
        """ Check of toxicity on russian"""
        toxicity_evaluation = self.text2toxicity_ru('Tебя зовут ' + string.title() + '?')
        if not self.silent:
            print('ru: ', string)
        if any(i > self.toxity_level for i in toxicity_evaluation[1:]):
            return True
        else:
            return False


"""
text2 = pd.read_excel('/content/drive/MyDrive/кейсы с именами.xlsx')
checker = Safe_naming(toxity_level=0.5, tokenizer_ru=tokenizer_ru, model_ru=model_ru, text2toxicity_en=text2toxicity_en, silent =True)
text2['toxity_level=0.5'] = text2.text.apply(lambda x: checker.check_safety(x))
checker = Safe_naming(toxity_level=0.1, tokenizer_ru=tokenizer_ru, model_ru=model_ru, text2toxicity_en=text2toxicity_en, silent =True)
text2['toxity_level=0.1'] = text2.text.apply(lambda x: checker.check_safety(x))
checker = Safe_naming(toxity_level=0.05, tokenizer_ru=tokenizer_ru, model_ru=model_ru, text2toxicity_en=text2toxicity_en, silent =True)
text2['toxity_level=0.05'] = text2.text.apply(lambda x: checker.check_safety(x))
checker = Safe_naming(toxity_level=0.01, tokenizer_ru=tokenizer_ru, model_ru=model_ru, text2toxicity_en=text2toxicity_en, silent =True)
text2['toxity_level=0.01'] = text2.text.apply(lambda x: checker.check_safety(x))

checker = Safe_naming_prompt_q_title(toxity_level=0.05, tokenizer_ru=tokenizer_ru, model_ru=model_ru, text2toxicity_en=text2toxicity_en, silent=False)
checker = Safe_naming_prompt_q(toxity_level=0.05, tokenizer_ru=tokenizer_ru, model_ru=model_ru, text2toxicity_en=text2toxicity_en, silent=False)


print (checker.check_safety('nihuya sebe'))

"""
