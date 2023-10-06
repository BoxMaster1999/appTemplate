import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


class SafeNaming:
    """This class contains functions to check if the typed sentence is toxic or just contains a personal name"""

    def __init__(self, toxity_level=0.5, silent=False):
        """Constructor"""
        # Dowload all models online from Huggingface everytime
        path_to_rus_model = 'cointegrated/rubert-tiny-toxicity'
        path_to_eng_model = 'citizenlab/distilbert-base-multilingual-cased-toxicity'
        self.model_ru = AutoModelForSequenceClassification.from_pretrained(path_to_rus_model)
        self.tokenizer_ru = AutoTokenizer.from_pretrained(path_to_rus_model)
        self.text2toxicity_en = pipeline("text-classification", model=path_to_eng_model)
        print('model loading succeed')
        self.silent = silent
        self.toxity_level = toxity_level

    def text2toxicity_ru(self, text):
        """ Calculate  a vector of toxicity aspects on russian"""
        with torch.no_grad():
            inputs = self.tokenizer_ru(text, return_tensors='pt', truncation=True, padding=True)
            proba = torch.sigmoid(self.model_ru(**inputs).logits).numpy()
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

    def full_check_for_all_russian(self, string) -> bool:
        if self.check_russian(string):  # проверка как русского слова
            return True
        if self.check_english(
                self.transliteration_sound_ru_en(string)):  # звуковая транслитерация и проверка как аглийского слова
            return True
        if self.check_english(
                self.transliteration_sumbol_ru_en(string)):  # символьная транслитерация и проверка как аглийского слова
            return True
        return False

    def full_check_for_all_english(self, string) -> bool:
        if self.check_english(string):  # проверка как аглийского слова
            return True

        if self.check_russian(
                self.transliteration_sound_en_ru(string)):  # звуковая транслитерация и проверка как русского слова
            return True

        if self.check_russian(
                self.transliteration_sumbol_en_ru(string)):  # символьная транслитерация и проверка как русского слова
            return True

        return False

    def check_safety(self, string: str) -> bool:

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

safe_naming_checker = SafeNaming()
print(safe_naming_checker.check_safety('Пидор'))
