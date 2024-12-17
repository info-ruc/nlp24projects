import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import shap

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

from utils import Ielts_Preprocess

def ielts_score(prediction):
    if prediction < 0:
        return 0.0
    elif prediction > 9:
        return 9.0
    else:
        return round(prediction * 2) / 2  # 四舍五入到最近的0.5分

class MyError(Exception):
    def __init__(self, type, message):
        super().__init__(message)
        self.type = type
        self.message = message

    def __str__(self):
        if self.type == 'format':
            return f"Format Using Error: {self.message}"


class IeltsModel():
    # 导入数据
    def __init__(self, format, *args, **kwargs):
        self.format = format
        if format == 'single':
            self.data = pd.DataFrame({'problem': [kwargs.get('problem', [])], 'sub_article': [kwargs.get('article', [])]})
        elif format == 'dataset':
            self.data = kwargs.get('data', pd.DataFrame())
        else:
            raise MyError('format', "Only 'single' or 'dataset' as parameter format is allowed!")
        
        self.path_name = kwargs.get('path', 'temp')

        self.pre_tool = Ielts_Preprocess()
        
        # 文本向量化
        self.article_tokenizer = BertTokenizer.from_pretrained('E:/vscode-L/PYTHON/NLP/bert-base-uncased/')
        self.article_model = BertModel.from_pretrained('E:/vscode-L/PYTHON/NLP/bert-base-uncased/')

        # 摘要提取
        self.summary_tokenizer = AutoTokenizer.from_pretrained("E:/vscode-L/PYTHON/NLP/sshleifer/distilbart-cnn-12-6")
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained("E:/vscode-L/PYTHON/NLP/sshleifer/distilbart-cnn-12-6")

        # 读取预测模型
        with open('E:/vscode-L/PYTHON/NLP/IeltsScoring/model_tr.pkl', 'rb') as f:
            self.model1 = pickle.load(f)

        with open('E:/vscode-L/PYTHON/NLP/IeltsScoring/model_cc.pkl', 'rb') as f:
            self.model2 = pickle.load(f)
        
        with open('E:/vscode-L/PYTHON/NLP/IeltsScoring/model_lr.pkl', 'rb') as f:
            self.model3 = pickle.load(f)
        
        with open('E:/vscode-L/PYTHON/NLP/IeltsScoring/model_ga.pkl', 'rb') as f:
            self.model4 = pickle.load(f)
        
        # 模型解释性分析
        self.explainer1 = shap.explainers.Tree(self.model1)
        self.explainer2 = shap.explainers.Tree(self.model2)
        self.explainer3 = shap.explainers.Tree(self.model3)
        self.explainer4 = shap.explainers.Tree(self.model4)
    
    def append_data(self, problem, article):
        self.data = pd.DataFrame({'problem': [problem], 'sub_article': [article]})

    # 数据预处理+特征工程，导出数据集，返回df
    def preprocess(self):
        print("Start to calculate important features.")
        # 处理分段
        def get_subParagraph(text):
            paragraphs = text.splitlines()
            result = ' '.join(paragraphs)
            num_paragraphs = len(paragraphs)
            return result, num_paragraphs
        
        self.data['article'] = [0]*len(self.data)
        self.data['subParagraph'] = [0]*len(self.data)

        for i in range(len(self.data)):
            text = self.data['sub_article'][i]
            article, subParagraph = get_subParagraph(text)
            self.data['article'][i] = article
            self.data['subParagraph'][i] = subParagraph
        
        results = self.data['article'].apply(self.pre_tool.process_article)
        results_df = pd.DataFrame(list(results))
        self.data = pd.concat([self.data, results_df], axis=1)
        
        self.data.to_csv(f'E:/vscode-L/PYTHON/NLP/IeltsScoring/data_{self.path_name}_pre.csv', index=False)
        print(f"Already save pre-dataset to E:/vscode-L/PYTHON/NLP/IeltsScoring/data_{self.path_name}_pre.csv")
        
        return self.data
    
    # 文章和题目向量化
    def bert_process(self):
        print("Start to transform texts into vectors.")

        self.article_model.eval()
        self.summary_model.eval()

        # 分割文章并获取Pooling结果
        def get_pooled_output(text):
            paragraphs = text.splitlines()
            encoded_input = self.article_tokenizer(paragraphs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            
            with torch.no_grad():
                outputs = self.article_model(**encoded_input)
            
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            pooled_output = np.mean(cls_embeddings, axis=0) # 取平均值池化
            return pooled_output

        def process_articles(df): 
            pooled_outputs = []
            for index, row in df.iterrows():
                article = row['sub_article']
                pooled_output = get_pooled_output(article)
                pooled_outputs.append(pooled_output)
                
            outputs = pd.DataFrame(pooled_outputs)
            return outputs
        
        self.pooled_outputs = process_articles(self.data)
        self.pooled_outputs.columns = [f'pooled_article_{i}' for i in range(self.pooled_outputs.shape[1])]
        
        # 全文向量化
        def get_bert_embeddings(texts):
            embeddings = []
            for text in texts:
                inputs = self.article_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.article_model(**inputs)

                cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
                embeddings.append(cls_embedding)
            
            return embeddings

        embeddings = get_bert_embeddings(self.data['article'].tolist())
        self.embedding_outputs = pd.DataFrame(embeddings)
        self.embedding_outputs.columns = [f'articleEB_{i}' for i in range(self.embedding_outputs.shape[1])]

        # 题目向量化
        embeddings = get_bert_embeddings(self.data['problem'].tolist())
        self.problem_outputs = pd.DataFrame(embeddings)
        self.problem_outputs.columns = [f'problemEB_{i}' for i in range(self.problem_outputs.shape[1])]

        # 摘要提取
        texts = self.data['article'].to_list()
        summary = []
        for text in texts:
            inputs = self.summary_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.summary_model.generate(**inputs, max_length=100, min_length=80)
            summary.append(self.summary_tokenizer.decode(outputs[0], skip_special_tokens=True))
        
        self.data['summary'] = summary
        embeddings = get_bert_embeddings(summary)
        self.summary_outputs = pd.DataFrame(embeddings)
        self.summary_outputs.columns = [f'summaryEB_{i}' for i in range(self.summary_outputs.shape[1])]

        print("Finish transforming tasks.")

        # 计算余弦相似度
        def get_sim(article, problem):
            sim = []
            for i in range(len(article)):
                vector1 = article.iloc[i,:].values.reshape(1, -1)
                vector2 = problem.iloc[i,:].values.reshape(1, -1)
                
                # 计算余弦相似度
                similarity = cosine_similarity(vector1, vector2)[0][0]
                sim.append(similarity)

            return sim
        
        self.embedding_outputs['sim'] = get_sim(self.embedding_outputs, self.problem_outputs)
        self.pooled_outputs['sim'] = get_sim(self.pooled_outputs, self.problem_outputs)
        self.summary_outputs['sim'] = get_sim(self.summary_outputs, self.problem_outputs)

        # 数据降维
        with open('E:/vscode-L/PYTHON/NLP/IeltsScoring/scaler_model.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('E:/vscode-L/PYTHON/NLP/IeltsScoring/pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)

        problem_reduced = pca.transform(scaler.transform(self.problem_outputs))
        self.problem_reduced = pd.DataFrame(problem_reduced, columns=[f'problemPCA_{i}' for i in range(problem_reduced.shape[1])])

        # 合并数据集
        cols = [False]*3 + [True] * (len(self.data.columns)-4) + [False]
        X = self.data.loc[:,cols]
        X['sim1'] = self.embedding_outputs['sim']
        X['sim2'] = self.pooled_outputs['sim']
        X['sim3'] = self.summary_outputs['sim']
        X = pd.concat([X, self.pooled_outputs.iloc[:,:-1], self.problem_reduced], axis=1)

        path = f'E:/vscode-L/PYTHON/NLP/IeltsScoring/data_{self.path_name}.csv'
        X.to_csv(path, index=False)
        print(f"Already save whole dataset to {path}")

        self.X = X
        return path

    # 从四个维度进行作文评分，并给出简要评价
    def predict_scores(self):
        # 打分
        pred = [self.model1.predict(self.X), self.model2.predict(self.X),
                self.model3.predict(self.X), self.model4.predict(self.X)]
        
        TR = [ielts_score(pred) for pred in pred[0]]
        CC = [ielts_score(pred) for pred in pred[1]]
        LR = [ielts_score(pred) for pred in pred[2]]
        GA = [ielts_score(pred) for pred in pred[3]]

        score =  (np.array(TR) + np.array(CC) + np.array(LR) + np.array(GA)) / 4
        score = [ielts_score(pred) for pred in score]

        # TODO: 基于shap给出四维评价
        def score_range(score):
            if score <= 4.5:
                return '0-4.5'
            if score <= 6.5:
                return '5-6.5'
            if score <= 8:
                return '7-8'
            else:
                return '8.5-9'
            
        def evaluation(score, errors, shap):
            eval_text = ''

            # 任务响应 
            tr_format = {'0': '文章尝试回答了问题，并且对观点有一定的阐述。然而，论证部分比较薄弱，且没有完整地响应题目，理由没有展开，缺少深入分析。建议在论证中加入更多具体的支持性例子、数据或研究结果，使论点更加有说服力。',
                         '1': '文章回答了题目中的问题，结构比较清晰并给出了具体的例子来支持论点，但文章中的论证和分析略显简单。为了提高任务回应的分数，可以在讨论正面和负面影响时，均提供一定的细节和论据，增加文章的深度，保证文章的平衡性。',
                         '2': '文章有效地回应了题目要求，提供了多方面的论点，但可以通过更多的不同角度的举例，使文章论点更有力、平衡。',
                         '3': '文章充分回应了题目要求，对问题进行了明确的回答。作者在开头明确表达了自己对于题目的立场，在正文部分用具体的例子来支持自己的观点，结论部分重申了观点，清晰地总结了全文的主旨。整体上来说，文章结构清晰，观点明确，论证有力。如需进一步改进，可以在某些段落中进一步扩展不同角度的讨论，尤其是具体的案例和措施可能会使论证更为深入。'}
            
            def tr_analysis(shap_values, score):
                value = 0
                for f, v in shap_values:
                    if f in ['sim1', 'sim2', 'sim3'] and abs(v) > abs(value):
                        value = v
                if score <= 4.5 or value <= -1e-2:
                    return '0'
                elif score <= 6.5 or value <= 0:
                    return '1'
                elif score <= 8 or value <= 1e-2:
                    return '2'
                else:
                    return '3'
                
            
            print(f"1. Task Response (任务回应) – 评分: {score[0]}")
            print(f"评价：{tr_format[tr_analysis(shap[0], score[0])]} \n")
            eval_text += f"1. Task Response (任务回应) – 评分: {score[0]}\n" + f"评价：{tr_format[tr_analysis(shap[0], score[0])]} \n"

            # 连贯 & 衔接
            cc_format = {'0-4.5': '文章中使用的论点和论据组织不当，推进不连贯，缺乏逻辑组织。建议重新组织行文，尝试将文章划分为引入观点、主体论证、结论三部分，用适当的衔接词、过渡句将观点串联起来。',
                         '5-6.5': '文章行文结构比较完整，有引言、主体段和结论，但在某些地方，衔接词的使用略显简单，段落之间有时过渡得较为生硬。可以在段落之间增加更多的连接词和过渡句，使文章的逻辑更加连贯，避免过度依赖简单的连接词。',
                         '7-8': '文章的结构比较清晰，作者通过段落的划分有条理地回答了问题，并且使用了一些衔接词，使得文章的逻辑清晰，行文顺畅，段落之间的转折和衔接较为自然。在某些地方可以使用更多不同的过渡词和句型，来增加文章的语言多样性。',
                         '8.5-9': '文章的结构清晰，逻辑性强。每一段都紧扣主题，且段落间衔接自然。论点的展开顺畅，每一段都通过合适的连接词和转折词引导到下一个论点。'}
            
            def cc_analysis(shap_values, score):
                cc_f = ['Flesch_Kincaid_score', 'Gunning_Fog_score', 'SMOG_score']
                value = 0
                keys = []

                for f, v in shap_values:
                    if f in cc_f and abs(v) > abs(value):
                        value = v
                    elif f == 'conjunctions' and v > 0:
                        keys.append(1)
                    elif f == 'conjunctions' and v < 0:
                        keys.append(-1)
                    elif f == 'cohesion' and v > 0:
                        keys.append(2)
                    elif f == 'cohesion' and v < 0:
                        keys.append(-2)

                text = ''
                if 1 in keys and 2 in keys:
                    text = '作者善于使用连接词和过渡句，'
                elif 1 in keys and -2 in keys:
                    text = '作者善于使用连接词和复合句，但过渡句不够明确，'
                elif -1 in keys and 2 in keys:
                    text = '作者善于使用过渡句来明确稳文章结构，但句式较为简单，'
                elif -1 in keys and -2 in keys:
                    text = '整体而言，句式结构和过渡句使用过于简单，不够清晰，'
                elif 1 in keys:
                    text = '作者善于使用连接词和复合句，'
                elif 2 in keys:
                    text = '作者善于使用过渡句，'
                elif -1 in keys:
                    text = '文章整体句式结构比较简单，'
                elif -2 in keys:
                    text = '文章似乎缺乏明确的过渡句和衔接，'
                
                if value > 0:
                    text += '同时，行文比较清晰，可读性强。'
                elif value < 0:
                    text += '同时，文章可读性略差，建议尝试合理搭配简单句和长难句，使得行文更流畅自然。'

                return text + cc_format[score_range(score)]

            print(f"2. Coherence and Cohesion (连贯与衔接) – 评分: {score[1]}")
            print(f"评价：{cc_analysis(shap[1], score[1])} \n")
            eval_text += f"\n2. Coherence and Cohesion (连贯与衔接) – 评分: {score[1]}\n" + f"评价：{cc_analysis(shap[1], score[1])} \n"

            # 词汇丰富
            lr_format = {'0-4.5': f'词汇使用不够丰富，建议增加词汇多样性，避免重复使用相同的单词；注意词汇的搭配和拼写，确保使用正确的表达。',
                         '5-6.5': f'文章用词比较准确，能够清楚传达作者的观点，但可以尝试用一些同义词或变通的表达方式避免重复，注意在使用一些高级词汇时，要确保搭配自然。',
                         '7-8': f'作者在文章中使用了一些高级词汇和短语，增强了文章的表达能力，展现出作者较强的词汇能力。虽然高端词汇运用较好，但在某些地方还可以加入更多简单的词汇进行平衡，避免过于复杂的句子结构影响可读性。',
                         '8.5-9': f'文章中词汇使用精准，能够精确表达观点。'}
            
            def lr_analysis(shap_values, score):
                keys = []
                for f, v in shap_values:
                    if f == 'important_words' and v > 1e-2:
                        keys.append('能够使用一些高级表达，对于文章评分属于加分项')
                    elif f == 'important_words' and v < -5e-3:
                        keys.append('使用的词汇比较基础，选择更高级的表达会取得更高的分数')
                    elif f == 'repalce' and v > 0:
                        keys.append('善于使用同义替换，使得文章的表达更丰富')
                    elif f == 'repalce' and v < 0:
                        keys.append('词汇使用存在同质化，可以尝试使用同义替换')
                if len(keys) > 0:
                    text = '作者' + '；'.join(keys) + '。整体而言，' + lr_format[score_range(score)]
                else:
                    text = '整体而言，' + lr_format[score_range(score)]
                return text
            
            print(f"3. Lexical Resource (词汇资源) – 评分: {score[2]}")
            print(f"评价：{lr_analysis(shap[2], score[2])} \n")
            eval_text += f"\n3. Lexical Resource (词汇资源) – 评分: {score[2]}\n" + f"评价：{lr_analysis(shap[2], score[2])} \n"

            # 语法
            ga_format = {'0-4.5': f'文章中存在多个语法错误，尤其是在句子结构、动词时态、主谓一致等方面。',
                         '5-6.5': f'句子结构相对多样，能使用一些复合句，基本表达清晰，但存在一些语法错误和不太准确的表达。可以多加注意语法的准确性，避免使用不自然或不合适的句型。',
                         '7-8': f'文章的语法结构总体准确，使用了多样的句型，且大多数句子的语法都是正确的。然而，有一些小的语法问题和可改进之处。',
                         '8.5-9': f'文章展示了丰富的语法结构和作者较高的语法水平，使用了丰富的句式结构，整体上没有明显错误，句子结构清晰，逻辑性强。'}
            
            def ga_analysis(shap_values, score):
                errors = []
                sentence = []
                for f, v in shap_values:
                    if 'spelling_errors' in f and v < -1e-3 and '拼写错误' not in errors:
                        errors.append('拼写错误')
                    elif f == 'tense_errors' and v < 0:
                        errors.append('时态错误')
                    elif f == 'plural_errors' and v < 0:
                        errors.append('单复数错误')
                    elif f == 'upper' and v < -1e-3:
                        errors.append('大小写错误')
                    elif f == 'complex_simple_ratio' and v > 5e-3:
                        sentence.append('复合句、长难句')
                    elif f == 'passive_ratio' and v > 1e-2:
                        sentence.append('被动语态')
                    
                text = ''
                if score < 8 and len(errors) > 0:
                    text = '文章存在' + '、'.join(errors) + '等多种语法错误，'
                if len(sentence) > 0:
                    text += '作者善于使用、搭配' + '和'.join(sentence) + '使得文章的语法结构更丰富。'
                elif score < 7:
                    text += '作者使用的句式结构可能比较单一。'
                
                if len(text) > 0:
                    return text + '\n     整体而言，' + ga_format[score_range(score)]
                else:
                    return ga_format[score_range(score)]

            print(f"4. Grammatical Range and Accuracy (语法多样与准确性) – 评分: {score[3]}")
            print(f"评价：{ga_analysis(shap[3], score[3])} \n")
            eval_text += f"\n4. Grammatical Range and Accuracy (语法多样与准确性) – 评分: {score[3]}\n" + f"评价：{ga_analysis(shap[3], score[3])} \n"

            # 输出错误信息
            if len(errors['spelling_errors']) > 0:
                print("单词拼写错误和修正如下：")
                eval_text += "单词拼写错误和修正如下：\n"

                cnt = 1
                for word, suggestions in errors['spelling_errors'].items():
                    print(f"{cnt}. 错误拼写：{word}，纠正猜测：{suggestions}")
                    eval_text += f"{cnt}. 错误拼写：{word}，纠正猜测：{suggestions}\n"
                    cnt += 1

            if len(errors['grammar_errors']) > 0:
                print("\n语法错误如下：")
                eval_text += "\n语法错误如下：\n"

                cnt = 1
                for error in errors['grammar_errors']:
                    print(f"{cnt}. 语法错误部分：{error['error_text']}")
                    print(f"   错误类型：{error['error_type']}，修正建议：{error['suggestions']}")
                    eval_text += f"{cnt}. 语法错误部分：{error['error_text']}\n" + f"   错误类型：{error['error_type']}，修正建议：{error['suggestions']}\n"
                    cnt += 1

            # 综合
            print(f"\n综合评分：{score[4]}")
            eval_text += f"\n综合评分：{score[4]}"

            return eval_text

        error_list = self.get_recommendations()

        shap_values_tr = self.explainer1(self.X)
        shap_values_cc = self.explainer2(self.X)
        shap_values_lr = self.explainer3(self.X)
        shap_values_ga = self.explainer4(self.X)

        eval_text_list = []
        for i in range(len(score)):
            if self.format == 'dataset':
                print(f"\n题目：{self.data.iloc[i, 0]}")
                print(f"文章：\n{self.data.iloc[i, 1]}\n")

            shap_tr = [(f,v) for f,v in zip(self.X.columns, shap_values_tr[i].values) if 'problem' not in f and 'article' not in f and abs(v) > 1e-4]
            shap_tr.sort(key=lambda x: abs(x[1]), reverse=True)

            shap_cc = [(f,v) for f,v in zip(self.X.columns, shap_values_cc[i].values) if 'problem' not in f and 'article' not in f and abs(v) > 1e-4]
            shap_cc.sort(key=lambda x: abs(x[1]), reverse=True)

            shap_lr = [(f,v) for f,v in zip(self.X.columns, shap_values_lr[i].values) if 'problem' not in f and 'article' not in f and abs(v) > 1e-4]
            shap_lr.sort(key=lambda x: abs(x[1]), reverse=True)

            shap_ga = [(f,v) for f,v in zip(self.X.columns, shap_values_ga[i].values) if 'problem' not in f and 'article' not in f and abs(v) > 1e-4]
            shap_ga.sort(key=lambda x: abs(x[1]), reverse=True)

            eval_text = evaluation([TR[i], CC[i], LR[i], GA[i], score[i]], error_list[i], [shap_tr, shap_cc, shap_lr, shap_ga])
            
            if self.format == 'dataset':
                eval_text = f"题目：{self.data.iloc[i, 0]}\n" + f"文章：\n{self.data.iloc[i, 1]}\n\n" + eval_text

            eval_text_list.append(eval_text)    

        return eval_text_list, TR, CC, LR, GA, score
    
    # 从语法、词汇等角度提出修改建议
    def get_recommendations(self):
        errors_list = []
        for i in range(len(self.data)):
            article = self.data.iloc[i, 2]
            erros = self.pre_tool.check_grammar(self.pre_tool.process_text(article))
            errors_list.append(erros)
        return errors_list
    
    # 批改结果导出
    def results_to_csv(self, eval_text_list, TR, CC, LR, GA, score):
        result = pd.DataFrame({'TR': TR,'CC': CC,'LR': LR,'GA': GA,'score': score,'evaluation': eval_text_list})
        result = pd.concat([self.data[['problem','sub_article']], result], axis=1)
        
        path = f'E:/vscode-L/PYTHON/NLP/IeltsScoring/result_{self.path_name}.csv'
        result.to_csv(path, index=False)
        print(f"Already save scores and evaluation to {path}")

        return path
    
    # 作文批改主函数
    def main(self):
        # 数据预处理
        self.preprocess()
        self.bert_process()
        # 预测和评价输出
        eval_text_list, TR, CC, LR, GA, score = self.predict_scores()
        # 导出批改结果，包括题目、文章、四维评分、修改建议
        self.results_to_csv(eval_text_list, TR, CC, LR, GA, score)

        return