import re
import textstat
import nltk
from nltk.corpus import stopwords, wordnet
from nltk import sent_tokenize, word_tokenize, pos_tag
import spacy
from collections import Counter
from spellchecker import SpellChecker
import language_tool_python
from autocorrect import Speller
import pickle

import warnings
warnings.filterwarnings('ignore')

class Ielts_Preprocess():
    def __init__(self, *args, **kwargs):
        self.nlp = spacy.load("en_core_web_sm")
        self.spell = SpellChecker()
        self.tool = language_tool_python.LanguageTool('en-US')

        with open("E:/vscode-L/PYTHON/NLP/IeltsScoring/ielts_dictionary.pkl", "rb") as f:
            self.score_dict = pickle.load(f)
        self.important_words = list(self.score_dict.keys())

        return
    
    def count_capitalization_errors(self, text):
        """统计大小写错误数量"""
        sentences = re.split(r'(?<=[.!?]) +', text)
        upper_count = 0
        for sentence in sentences:
            stripped_sentence = sentence.strip()
            if stripped_sentence:
                if not stripped_sentence[0].isupper():
                    upper_count += 1
                    
        return upper_count
    
    def determine_expected_tense(self, tokens):
        """根据句子中的时间词确定期望的时态"""
        for token in tokens:
            if token.dep_ == "nsubj":  # 找到主语
                if any(time_word in token.text.lower() for time_word in ["yesterday", "last week", "ago", "since","in the past", "last month", "last year", "the other day"]):
                    return "past"
                elif any(time_word in token.text.lower() for time_word in ["now", "always", "every", "usually", "often", "sometimes"]):
                    return "present"
                elif any(time_word in token.text.lower() for time_word in ["tomorrow", "next", "in the future"]):
                    return "future"
        return "unknown"
    
    def is_correct_tense(self, token, expected_tense):
        """检查动词时态是否符合预期"""
        if expected_tense == "past":
            return token.tag_ in ["VBD", "VBN"]  # 过去时和过去分词
        elif expected_tense == "present":
            return token.tag_ in ["VB", "VBZ", "VBG"]  # 现在时
        elif expected_tense == "future":
            return token.dep_ == "aux" and token.lemma_ == "will"  # 将来时
        return True

    def check_tense_errors(self, sentence):
        """检查句子中的时态错误"""
        tokens = self.nlp(sentence)  # 使用 spaCy 解析句子
        tense_errors = 0
        expected_tense = self.determine_expected_tense(tokens)
        
        for token in tokens:
            if token.dep_ == "ROOT":  # 找到句子的主要动词
                if not self.is_correct_tense(token, expected_tense):
                    tense_errors += 1
        return tense_errors

    # 拼写错误检测
    def spell_check(self, doc, text):
        misspelled = self.spell.unknown([token.text for token in doc if token.is_alpha])  # 仅检查字母字符

        spell = Speller()
        corrected_text = spell(text)
        
        original_words = text.split()
        corrected_words = corrected_text.split()
        errors = sum(1 for orig, corr in zip(original_words, corrected_words) if orig != corr)
        
        return len(misspelled), errors

    # 检查时态错误
    def tense_error_check(self, text):
        sentences = sent_tokenize(text)
        total_errors = 0
        for sentence in sentences:
            errors = self.check_tense_errors(sentence)
            total_errors += errors
        
        return total_errors

    # 检查单复数使用错误
    def plural_error_check(self, doc):
        errors = 0
        for token in doc:
            if token.dep_ == 'nsubj' and token.tag_ == 'NN':  # 单数名词
                for child in token.children:
                    if child.dep_ == 'ROOT':
                        if child.tag_ != 'VBZ':  # 单数主语应配合单三形式
                            errors += 1
            
            elif token.dep_ == 'nsubj' and token.tag_ == 'NNS':  # 复数名词
                for child in token.children:
                    if child.dep_ == 'ROOT':
                        if child.tag_ == 'VBZ':  # 复数主语不应使用单三形式
                            errors += 1
        return errors

    # 句型统计
    def sentence_type_statistics(self, doc):
        simple = 0
        conj = 0
        clause1 = 0
        clause2 = 0
        for sent in doc.sents:
            tokens = [token for token in sent]
            # 统计主语、谓语、连词、从句数量
            subjects = [token for token in tokens if token.dep_ == 'nsubj']
            predicates = [token for token in tokens if token.dep_ == 'ROOT']
            conjunctions = [token for token in tokens if token.dep_ == 'cc']
            ccomp = [token for token in tokens if token.dep_ == 'ccomp']
            acl = [token for token in tokens if token.dep_ == 'acl']
            
            if len(subjects) > 0 and len(predicates) > 0:
                if len(subjects) == 1 and len(predicates) == 1 and len(ccomp) + len(acl) == 0:
                    simple += 1  # 简单句
                if len(conjunctions) > 0:
                    conj += 1
                if len(ccomp) > 0:
                    clause1 += 1
                if len(acl) > 0:
                    clause2 += 1
        return simple,conj,clause1,clause2
    
    # 语法错误检查
    def check_grammar(self, text):
        punct = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
        translation_table = str.maketrans(punct, ' ' * len(punct))
        punct_text = text.translate(translation_table)

        # 拼写错误
        spelling_errors = {}
        words = punct_text.split()
        for word in words:
            if word not in self.spell:
                corrected = self.spell.candidates(word)

                if corrected is None:
                    spelling_errors[word] = None
                else:
                    spelling_errors[word] = list(corrected)  # 获取拼写建议

        # 检查语法错误
        grammar_matches = self.tool.check(text)
        grammar_errors = []
        
        for match in grammar_matches:
            if match.ruleId in ['MORFOLOGIK_RULE_EN_US', 'WHITESPACE_RULE','CONSECUTIVE_SPACES', 'UPPERCASE_SENTENCE_START']:
                continue

            error_info = {
                'error_text': match.context,  # 错误上下文
                'error_type': match.ruleId,  # 错误类型
                'suggestions': match.replacements,  # 建议替换
                'error_position': (match.offset, match.errorLength)  # 错误位置
            }
            grammar_errors.append(error_info)

        return {
            'spelling_errors': spelling_errors,
            'grammar_errors': grammar_errors
        }
    
    # 词汇多样性
    def TTR(self, words):
        num_words = len(words)
        unique_words = len(set(words))
        ttr = unique_words / num_words if num_words > 0 else 0  # 类型-令词比
        return ttr

    # 句子结构
    def sentence_complexity(self, sentences, words):
        num_sentences = len(sentences)
        avg_sentence_length = len(words) / num_sentences if num_sentences > 0 else 0
        longest_sentence = max(len(word_tokenize(sentence)) for sentence in sentences) if sentences else 0
        shortest_sentence = min(len(word_tokenize(sentence)) for sentence in sentences) if sentences else 0

        # 复杂句与简单句比例
        complex_sentences = sum(1 for sentence in sentences if any(pos.startswith('IN') for _, pos in pos_tag(word_tokenize(sentence))))
        complex_simple_ratio = complex_sentences / num_sentences if num_sentences > 0 else 0

        return avg_sentence_length, longest_sentence, shortest_sentence, complex_simple_ratio

    # 检查指代使用情况
    def pronoun_usage(self, text):
        pronouns = re.findall(r'\b(I|you|he|she|it|we|they|my|your|his|her|its|our|their)\b', text, re.IGNORECASE)
        return len(pronouns)

    # 检查段落衔接与过渡句
    def cohesion_analysis(self, text):
        transitions = ['however', 'moreover', 'furthermore', 'in addition', 'therefore', 'consequently']
        transition_count = sum(text.lower().count(t) for t in transitions)
        return transition_count

    # 主动语态与被动语态比例
    def voice_ratio(self, sentences):
        active_count = sum(1 for s in sentences if 'by' not in s and re.search(r'\b(are|is|was|were|be)\b', s))
        passive_count = len(sentences) - active_count
        return passive_count / len(sentences)

    # 情态动词使用情况
    def modal_verbs_usage(self, text):
        modal_verbs = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
        return sum(text.lower().count(m) for m in modal_verbs)

    # 可读性指数
    def readability_scores(self, text):
        fk_score = textstat.flesch_kincaid_grade(text) # Flesch-Kincaid可读性指数：评估文本的可读性和难度。
        gf_score = textstat.gunning_fog(text) # Gunning Fog指数：用于评估理解文本所需的教育年限。
        smog_score = textstat.smog_index(text) # SMOG指数：分析文本所需的教育年限以理解
        return fk_score, gf_score, smog_score

    # 句型比例
    def sentence_types(self, sentences):
        types = {'declarative': 0, 'interrogative': 0, 'exclamatory': 0}
        
        for sentence in sentences:
            if sentence.endswith('?'):
                types['interrogative'] += 1
            elif sentence.endswith('!'):
                types['exclamatory'] += 1
            else:
                types['declarative'] += 1

        total = len(sentences)
        return {k: v / total for k, v in types.items() if total > 0}

    # 检查修辞手法
    def identify_rhetorical_devices(self, text):
        metaphors = re.findall(r'\b(as if|like|as though|seems|appears|compared to)\b', text, re.IGNORECASE)  
        return len(metaphors)
    
    # 统计高级词汇
    def count_important_words(self, text):
        words = text.split()

        cnt = 0
        cnt_high = 0
        for word in words:
            if word in self.important_words:
                cnt += self.score_dict[word]
                if self.score_dict[word] >= 9:
                    cnt_high += 1

        return cnt, cnt_high

    def find_synonym_replacements(self, text):
        words = nltk.word_tokenize(text.lower())
        word_count = Counter(words)

        def get_synonyms(word):
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
            synonyms.discard(word) # 去除原词
            return synonyms

        # 统计同义替换的次数
        replacement_count = 0
        checked_words = set() # 用于避免重复检查同一个词

        for word in word_count:
            if word not in checked_words:
                checked_words.add(word)
                synonyms = get_synonyms(word)
                if synonyms:
                    for synonym in synonyms:
                        if synonym in word_count:
                            replacement_count += 1
                            checked_words.add(synonym)

        return replacement_count

    # 处理爬虫时产生的错误
    def process_text(self, text):
        text = re.sub(r'\.(?=\S)', '. ', text)
        text = text.replace('‘', "'").replace('’', "'")
        return text

    # 特征工程汇总
    def process_article(self, article):
        doc = self.nlp(article)  # 处理文章
        text = self.process_text(article)

        upper = self.count_capitalization_errors(article)
        spelling_errors, spelling_errors_auto = self.spell_check(doc, text)  # 拼写错误统计
        tense_errors = self.tense_error_check(article)  # 时态错误统计
        plural_errors = self.plural_error_check(doc)  # 复数错误统计
        simple, conj, clause1, clause2 = self.sentence_type_statistics(doc)  # 句子类型统计

        sentences = sent_tokenize(text)
        words = nltk.word_tokenize(text.lower())
        
        ga = self.check_grammar(text)
        ttr = self.TTR(words)
        avg_sentence_length, longest_sentence, shortest_sentence, complex_simple_ratio = self.sentence_complexity(sentences, words)
        pronouns = self.pronoun_usage(text)
        cohesion = self.cohesion_analysis(text)
        passive_ratio = self.voice_ratio(sentences)
        modal = self.modal_verbs_usage(text)
        fk_score, gf_score, smog_score = self.readability_scores(text)
        '''
        type_cnt = sentence_types(sentences)
        declarative = type_cnt['declarative']
        interrogative = type_cnt['interrogative']
        exclamatory = type_cnt['exclamatory']
        '''
        metaphor = self.identify_rhetorical_devices(text)

        important_words, advanced_words = self.count_important_words(text)
        replace = self.find_synonym_replacements(text)
        
        return {
            'upper': upper,
            'spelling_errors': spelling_errors,
            'spelling_errors_auto': spelling_errors_auto,
            'spelling_errors_min': min(spelling_errors, spelling_errors_auto),
            'tense_errors': tense_errors,
            'plural_errors': plural_errors,
            'simple': simple,
            'conjunctions': conj,
            'ccomp': clause1,
            'acl': clause2,

            'TTR': ttr, # 词汇多样性

            'avg_sentence_length': avg_sentence_length,
            'longest_sentence': longest_sentence,
            'shortest_sentence': shortest_sentence,
            'complex_simple_ratio': complex_simple_ratio,

            'pronouns': pronouns, # 代词使用
            'cohesion': cohesion, # 过渡句
            'passive_ratio': passive_ratio, # 被动语态
            'modal_verbs_usage': modal, # 情态动词使用

            # 文章可读性/复杂性指数
            'Flesch_Kincaid_score': fk_score,
            'Gunning_Fog_score': gf_score,
            'SMOG_score': smog_score,
            
            'metaphor': metaphor, # 修辞手法

            'grammar_error': len(ga['grammar_errors']), # 语法错误总数

            'important_words': important_words,
            'advanced_words': advanced_words,

            'replace': replace # 使用同义替换的次数
        }

