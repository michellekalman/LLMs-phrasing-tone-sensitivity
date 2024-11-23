"""import evaluate
from nltk.translate.bleu_score import sentence_bleu
import warnings
from transformers import logging
from evaluate import load
from evaluate import load"""
from datasets import load_dataset
from itertools import islice
import tiktoken

"""
Evaluation
"""
"""
rouge = evaluate.load("rouge")


def eval_using_rouge(predictions, references):
    return rouge.compute(predictions=predictions, references=references)


def eval_using_bleu(predictions, references):
    predictions = predictions[0].split(" ")
    references = [sentence.split(" ") for sentence in references[0].split("\n")]
    return sentence_bleu(references, predictions)


warnings.filterwarnings("ignore")
logging.set_verbosity_error()

bertscore = load("bertscore")


def eval_using_bert(predictions, references):
    return bertscore.compute(predictions=predictions, references=references, lang="en")


meteor = evaluate.load('meteor')


def eval_using_meteor(predictions, references):
    return meteor.compute(predictions=predictions, references=references)

"""
"""
Create the dataset
"""

hesum = load_dataset('biunlp/HeSum')

model = "gpt-4-turbo"
encoding = tiktoken.encoding_for_model(model)


def not_too_long(data):
    if len(encoding.encode((data['article']))) < 4096 and len(data['article'].split('\n')) < 10:
        return True
    return False


hesum["test"] = hesum["test"].filter(not_too_long).shuffle().select(range(100))

politeness_levels = 8


"""
Create input prompts
"""

question_prompts = ["תסכם את המאמר הזה במשפט אחד יא זבל!", "אתה חייב לסכם את המאמר הבא במשפט אחד.", "אתה תסכם את המאמר הבא במשפט אחד.", "אתה נדרש לסכם את המאמר הבא במשפט אחד.", "סכם את המאמר הבא במשפט אחד. ", "סכם את המאמר הבא במשפט אחד, בבקשה.", "אשמח אם תוכל בבקשה לסכם את המאמר הבא במשפט אחד.", " אם זו לא טרחה גדולה מדי בשבילך, תסכם את המאמר הבא במשפט אחד, בבקשה."]
question_prompts = [prompt + "\n" for prompt in question_prompts]


def create_prompt(article, politeness_level):
    return question_prompts[politeness_level] + article


"""
Experiment
"""


def save_to_file(input_file, expected_output_file, dataset, articles_name, summaries_name, politeness_levels):
    for politeness_level in range(politeness_levels):
        input_file.write(f"politeness level: {str(politeness_level)}\n")
        expected_output_file.write(f"politeness level: {str(politeness_level)}\n")
        for data_idx in range(len(dataset[articles_name])):
            article = dataset[articles_name][data_idx]
            prompt = create_prompt(article, politeness_level)
            reference = [dataset[summaries_name][data_idx]]
            input_file.write(f"article {data_idx}: \n{prompt}\n")
            expected_output_file.write(f"article {data_idx}: \n{reference}\n")


with open("sum_input_heb.txt", "w", encoding="utf-8") as input_file:
    with open("sum_expected_output_heb.txt", "w", encoding="utf-8") as expected_output_file:
        save_to_file(input_file, expected_output_file, hesum["test"], "article", "summary", politeness_levels)
