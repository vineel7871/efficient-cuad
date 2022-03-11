from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json, torch

from transformers.data.processors.squad import SquadV2Processor, squad_convert_examples_to_features

# tokenizer = AutoTokenizer.from_pretrained("marshmellow77/roberta-base-cuad", use_fast=False)
tokenizer = AutoTokenizer.from_pretrained("MariamD/distilbert-base-uncased-finetuned-legal_data", use_fast=False)

# model = AutoModelForQuestionAnswering.from_pretrained("MariamD/distilbert-base-uncased-finetuned-legal_data")

data_path = "../data/CUAD_v1.json"

sample_path = "../data/cuad_sample.json"

# with open(sample_path,'r') as fobj:
#     data = json.loads(fobj.read())
#     data = data["data"]

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


def convert_squad_features_into_classification_features():
    pass

if __name__ == '__main__':  

    processor = SquadV2Processor()

    examples = processor.get_train_examples("data", filename="CUADv1.json")

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=True,
        return_dataset="pt",
        threads=4,
    )

    torch.save({"features": features, "dataset": dataset, "examples": examples}, ".\data\cached_train.json")

    # with open(".\data\sample_cached_1.json", 'w') as wobj:
    #     wobj.write(json.dumps({"features": features, "dataset": dataset, "examples": examples}))
