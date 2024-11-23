import evaluate
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
import warnings
from transformers import logging
from evaluate import load

print("loading evaluations")

rouge = evaluate.load("rouge")

politeness_levels = 8


def eval_using_length(predictions, references):
    pred_len = sum(len(prediction) for prediction in predictions)
    ref_len = sum(len(reference) for reference in references)
    return abs(pred_len-ref_len)/ref_len*100


def eval_using_rouge(predictions, references):
    return rouge.compute(predictions=predictions, references=references)


def eval_using_bleu(predictions, references):
    predictions = predictions[0].split(" ")
    references = [sentence.split(" ") for sentence in references[0].split("\n") if sentence]
    return sentence_bleu(references, predictions)


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

bertscore = load("bertscore")


def eval_using_bert(predictions, references):
    return bertscore.compute(predictions=predictions, references=references, lang="en")


meteor = evaluate.load('meteor')


def eval_using_meteor(predictions, references):
    return meteor.compute(predictions=predictions, references=references)


def compare_all(compare_func):
    with open("experiments/sum/heb/output/3.5/sum_output_3.5_heb.txt", "r", encoding="utf-8") as output_file:
        with open("experiments/sum/heb/expected_output/sum_expected_output_heb.txt", "r", encoding="utf-8") as expected_output_file:
            c = 0
            output_file_lines = output_file.readlines()
            expected_output_file_lines = expected_output_file.readlines()
            expected_output_file_idx = 0
            results = []
            output_line_idx = 0
            while output_line_idx < len(output_file_lines):
                if output_file_lines[output_line_idx].startswith("politeness level"):
                    results.append([])
                    output_line_idx += 1
                    expected_output_file_idx += 1
                else:
                    c += 1
                    print(f"starting {c}/{politeness_levels}00")
                    expected_output_file_idx += 1
                    output_line_idx += 1
                    results[-1].append(compare_func([output_file_lines[output_line_idx]], [expected_output_file_lines[expected_output_file_idx]]))
                    output_line_idx += 1
                    expected_output_file_idx += 1
    return results

print("running experiment")

"""
compare_all(eval_using_meteor), compare_all(eval_using_bert),
               compare_all(eval_using_rouge) compare_all(eval_using_bleu)
"""


def plot(data, ylabel):
    print(data)
    plt.figure()
    plt.plot(data)
    plt.xlabel("Politeness")
    plt.ylabel(ylabel)
    plt.title(f"Line Plot of {ylabel} Over Politeness")
    # plt.legend()
    plt.show()


def BERT_results():
    for result in [compare_all(eval_using_bert)]:
        print(result)
        p_level_precision = [0] * politeness_levels
        p_level_recall = [0] * politeness_levels
        p_level_f1 = [0] * politeness_levels
        for i in range(politeness_levels):
            p_level_precision[i] = np.average([precision for precision in result[0][i]['precision']])
            p_level_recall[i] = np.average([precision for precision in result[0][i]['recall']])
            p_level_f1[i] = np.average([precision for precision in result[0][i]['f1']])
        plot(p_level_recall, "BERT- precision")
        plot(p_level_precision, "BERT- recall")
        plot(p_level_f1, "BERT- f1")


def BLEU_results():
    result = compare_all(eval_using_bleu)
    print(result)
    p_level_bleu = [0] * politeness_levels
    for i in range(politeness_levels):
        p_level_bleu[i] = np.average(result[i])
    plot(p_level_bleu, "BLEU")


def rouge_results():
    for result in [compare_all(eval_using_rouge)]:
        print(result)
        p_level_rouge1 = [0] * politeness_levels
        p_level_rouge2 = [0] * politeness_levels
        p_level_rougeL = [0] * politeness_levels
        p_level_rougeLsum = [0] * politeness_levels
        for i in range(politeness_levels):
            p_level_rouge1[i] = np.average([precision['rouge1'] for precision in result[i]])
            p_level_rouge2[i] = np.average([precision['rouge2'] for precision in result[i]])
            p_level_rougeL[i] = np.average([precision['rougeL'] for precision in result[i]])
            p_level_rougeLsum[i] = np.average([precision['rougeLsum'] for precision in result[i]])
        plot(p_level_rouge1, "rouge1")
        plot(p_level_rouge2, "rouge2")
        plot(p_level_rougeL, "rougeL")
        plot(p_level_rougeLsum, "rougeLsum")


def meteor_results():
    for result in [compare_all(eval_using_meteor)]:
        print(result)
        p_level_meteor = [0] * politeness_levels
        for i in range(politeness_levels):
            p_level_meteor[i] = np.average([precision['meteor'] for precision in result[i]])
        plot(p_level_meteor, "meteor")


def length_results():
    for result in [compare_all(eval_using_length)]:
        print(result)
        p_level_length = [0] * politeness_levels
        for i in range(politeness_levels):
            p_level_length[i] = np.average([precision for precision in result[i]])
        plot(p_level_length, "length")


length_results()

print("finished")
