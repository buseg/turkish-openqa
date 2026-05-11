import argparse
import json
import os
from statistics import mean

import jsonlines
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def convert(input_data):
    qid2answers = {}
    for item in input_data:
        qid = item["id"]
        lang = item["lang"]
        answers = item["answers"]
        qid = qid.replace(f"-{lang}", "")
        if qid not in qid2answers:
            qid2answers[qid] = []
        qid2answers[qid] += answers

    for item in input_data:
        qid = item["id"]
        lang = item["lang"]
        qid = qid.replace(f"-{lang}", "")
        item['answers'] = qid2answers[qid]
    return input_data


def evaluate_top_k_hit(results, gt_answers, topk=10):
    per_lang = {}
    for item in tqdm(results):
        q_id = item["id"]
        lang = item["lang"]
        per_lang.setdefault(lang, {"count": 0, "hit": 0})
        ctxs = item["ctxs"]

        if q_id not in gt_answers:
            continue

        answers = gt_answers[q_id]

        span_answers = []
        # Skip yes/no examples during XOR-Retrieve evaluations
        for answer in answers:
            if answer not in ["yes", "no"]:
                span_answers.append(answer)
        if len(span_answers) == 0:
            continue

        per_lang[lang]["count"] += 1

        # concat_string_tokens = []
        # for ctx_text in ctxs[:topk]:
        #     tokenized_text = word_tokenize(ctx_text)
        #     concat_string_tokens += tokenized_text
        # concat_string = " ".join(concat_string_tokens)
        concat_string = " ".join(ctxs[:topk])
        hit = False
        for answer in span_answers:
            if answer in concat_string:
                hit = True
        if hit is True:
            per_lang[lang]["hit"] += 1

    final_results = {lang: result for lang,
                                      result in per_lang.items() if result["count"] > 0}

    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",
                        default=None, type=str)
    parser.add_argument("--pred_file",
                        default=None, type=str)
    parser.add_argument("--multi_lang", action="store_true")
    parser.add_argument("--topk", default=100, type=int)

    args = parser.parse_args()

    input_data = read_jsonlines(args.data_file)
    if args.multi_lang:
        input_data = convert(input_data)
    qid2answers = {item["id"]: item["answers"] for item in input_data if item["lang"] not in ['ar', 'bn', 'en', 'fi', 'ja', 'ko', 'ru', 'te']}
    qid2lang = {item["id"]: item["lang"] for item in input_data if item["lang"] not in ['ar', 'bn', 'en', 'fi', 'ja', 'ko', 'ru', 'te']}

    if "pids" in args.pred_file:
        predictions = read_jsonlines(args.pred_file)

        def load_corpus(corpus_file):
            import csv
            corpus = {}
            normalize = "all_w100.tsv" in corpus_file
            if normalize:
                import unicodedata
            with open(corpus_file, encoding='utf-8') as fIn:
                reader = csv.reader(fIn, delimiter="\t")
                for row in tqdm(reader):
                    if not row[0] == "id":
                        corpus[row[0]] = {
                            "title": row[2] if not normalize else unicodedata.normalize('NFC', row[2]),
                            "text": row[1]
                        }
            return corpus

        corpus = load_corpus(os.path.join('data/XOR-Full', "all_w100.tsv"))
        xor_output_prediction_format = []
        for example in predictions:
            qid, pids = example['qid'], example['pids']
            if qid not in qid2lang:
                continue
            ctxs = []
            for docid in pids:
                ctxs.append(corpus[docid]["text"])
            xor_output_prediction_format.append({"id": qid, "lang": qid2lang[qid], "ctxs": ctxs})
        predictions = xor_output_prediction_format
    else:
        predictions = json.load(open(args.pred_file))

    # convert input open-domain data into the qid2answer dictionary
    topk = args.topk
    print("Evaluating R@{}".format(topk))
    pred_per_lang_results = evaluate_top_k_hit(
        predictions, qid2answers, topk)
    avg_scores = []
    for lang in pred_per_lang_results:
        print("performance on {0} ({1} examples)".format(lang, pred_per_lang_results[lang]["count"]))
        per_lang_score = (pred_per_lang_results[lang]["hit"] / pred_per_lang_results[lang]["count"]) * 100
        print(per_lang_score)

        avg_scores.append(per_lang_score)

    print("Final macro averaged score: ")
    print(mean(avg_scores))


if __name__ == "__main__":
    main()
