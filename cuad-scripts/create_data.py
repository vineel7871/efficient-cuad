from logging import Logger
from urllib.parse import parse_qs
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json, torch
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from transformers import TrainingArgument
from transformers import Trainer
from transformers.data import load_metric
import tensorflow as tf

from transformers.data.processors.squad import SquadExample, SquadFeatures, SquadV2Processor, squad_convert_example_to_features, squad_convert_examples_to_features
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, TruncationStrategy
from transformers.utils.dummy_pt_objects import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("marshmellow77/roberta-base-cuad", use_fast=False)
tokenizer = AutoTokenizer.from_pretrained("MariamD/distilbert-base-uncased-finetuned-legal_data", use_fast=False)

# model = AutoModelForQuestionAnswering.from_pretrained("MariamD/distilbert-base-uncased-finetuned-legal_data")
model = TFAutoModelForSequenceClassification.from_pretrained(
    "MariamD/distilbert-base-uncased-finetuned-legal_data")

data_path = "../data/CUAD_v1.json"

sample_path = "../data/cuad_sample.json"

def get_data(path):
    with open(path,'r') as fobj:
        data = json.loads(fobj.read())
        data = data["data"]
    return data

# for x in data:
#     title = x['title']
#     for para in x['paragraphs']:
#         context = para['context']

#         for q in para["qas"]:

### Setting hyperparameters
max_seq_length = 512
doc_stride = 256
n_best_size = 1
max_query_length = 64
max_answer_length = 512
do_lower_case = False
null_score_diff_threshold = 0.0
batch_size = 32

def getClassificationFeatures(features:SquadFeatures):
    data = []
    for feature in features:
        data.append({"input_ids": feature.input_ids,
               "attention_mask": feature.attention_mask,
               "labels":feature.is_impossible})
    return data


# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def data_generator(path):

    input_data = get_data(path)
    examples = []
    is_training = False
    clf_features = []
    for entry in tqdm(input_data):
        title = entry["title"]
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position_character = None
                answer_text = None
                answers = []

                is_impossible = qa.get("is_impossible", False)
                if is_impossible:
                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    features = squad_convert_example_to_features(
                        example, max_seq_length, doc_stride, max_query_length, "max_length", is_training)

                    for f in features:
                        clf_features.append([{"input_ids": f.input_ids,
                                                "attention_mask": f.attention_mask}],
                                                [0])

                else:
                    answers = qa["answers"]
                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answers[0]['text'],
                        start_position_character=answers[0]["answer_start"],
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                    features = squad_convert_example_to_features(
                        example, max_seq_length, doc_stride, max_query_length, "max_length", is_training)

                    for f in features:
                        text = ' '.join(map(lambda x : f.token_to_orig_map[x],f.input_ids))
                        text = text.replace("[SEP]"," ")
                        for answer in answers:
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(answer['text']))
                            if cleaned_answer_text in text:
                                clf_features.append([{"input_ids": f.input_ids,
                                                        "attention_mask": f.attention_mask}],[1])
                            else:
                                clf_features.append([{"input_ids": f.input_ids,
                                                     "attention_mask": f.attention_mask}],
                                                     [0])
                for f in clf_features:
                    yield f


if __name__ == '__main__':

    train_features_generator = data_generator(
        "/Users/apple/Desktop/cognizer/python/cuad-demo/cuad-data/train_separate_questions.json")

    test_features_generator = data_generator(
        "/Users/apple/Desktop/cognizer/python/cuad-demo/cuad-data/test.json")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics="accuracy",
    )

    model.fit(train_features_generator, validation_data=test_features_generator, epochs=3)

    model.save_pretrained("my_classification_model")


