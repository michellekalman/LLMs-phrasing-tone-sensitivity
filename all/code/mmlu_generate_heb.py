from datasets import load_dataset
import random
from transformers import pipeline


"""
Create the dataset
"""


def not_too_long(data):
    if len(data['question']) > 40:
        return False
    for choice in data['choices']:
        if len(choice) > 40:
            return False
    return True


translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-he", max_length=900)


def translate(example):
    try:
        example['question'] = translator(example['question'])[0]['translation_text']
        example['choices'] = [translator(choice)[0]['translation_text'] for choice in example['choices']]
        return example
    except Exception as e:
        print(f"Exception during translate: {e}")
        return "fail"


politeness_levels = 8


"""
Create input prompts
"""

question_prompts = ["תענה על השאלות האלה יא זבל!", "אתה חייב לענות על השאלות הבאות", "אתה תענה על השאלות הבאות",
                    "אתה נדרש לענות על השאלות הבאות.", "ענה על השאלות הבאות.", "ענה על השאלות הבאות, בבקשה.",
                    "אשמח אם תוכל בבקשה לענות על השאלות הבאות.",
                    "אם זו לא טרחה גדולה מדי בשבילך, תענה על השאלות הבאות, בבקשה"]
format_prompts = ["שלא תעז לכתוב את התשובות בפורמט שהוא לא הפורמט הבא! ", "אתה חייב לכתוב את התשובות אך ורק בפורמט הבא:", "אתה תכתוב את התשובות בפורמט הבא:", "אתה נדרש לכתוב את התשובות רק בפורמט הבא:", "כתוב את התשובות בפורמט הבא:", "כתוב את התשובות בפורמט הבא:", "כתוב את התשובות בפורמט הבא, בבקשה:", "כתוב את התשובות בפורמט הבא, בבקשה:"]
question_prompts = [prompt + "\n" for prompt in question_prompts]
format_prompts = [prompt + "\n1. A" for prompt in format_prompts]


def format_mmlu_question(question, choices):
    formatted_question = f"{question}\n"
    for i, choice in enumerate(choices):
        formatted_question += f"\t{chr(65+i)}. {choice}\n"
    return formatted_question


def format_mmlu_multiple_questions(question_list, choices_list, politeness_level, k):
    prompt = question_prompts[politeness_level]
    for question_idx in range(len(question_list)):
        question = question_list[question_idx]
        choices = choices_list[question_idx]
        prompt += f"{question_idx+1}. {format_mmlu_question(question, choices)}"
    prompt += format_prompts[politeness_level]
    for i in range(2, k+1):
        letter_example = random.choice(["A", "B", "C", "D"])  # randomness is used in order to make it clear that this is an example and prevent bias
        prompt += f"\n{i}. {letter_example}"
    return prompt


choice_index = {"A": "0", "B": "1", "C": "2", "D": "3"}

"""
Experiment
"""


def save_to_file(input_file, expected_output_file, dataset, politeness_level, k):
    for example_idx in range(0, len(dataset["test"]), k):
        example = dataset["test"][example_idx:example_idx + k]
        formatted_input = format_mmlu_multiple_questions(example["question"], example["choices"], politeness_level, k)
        true_answer = example["answer"]
        input_file.write(f"questions from {example_idx}: \n" + formatted_input + "\n")
        expected_output_file.write(f"questions from {example_idx}: \n" + str(true_answer) + "\n")


def do_experiment(input_file, expected_output_file, dataset, politeness_levels):
    k = 3
    for politeness_level in range(politeness_levels):
        input_file.write("politeness level: " + str(politeness_level) + "\n")
        expected_output_file.write("politeness level: " + str(politeness_level) + "\n")
        save_to_file(input_file, expected_output_file, dataset, politeness_level, k)


with open("mmlu_input_heb_2.txt", "w", encoding="utf-8") as input_file:
    with open("mmlu_expected_output_heb_2.txt", "w", encoding="utf-8") as expected_output_file:
        for option in ["all"]:
            input_file.write(f"Subject: {option}\n")
            expected_output_file.write(f"Subject: {option}\n")
            mmlu = load_dataset("cais/mmlu", option)
            mmlu["test"] = mmlu["test"].filter(not_too_long).shuffle().select(range(200))
            mmlu_translated = mmlu.map(translate)
            mmlu_translated["test"] = mmlu_translated["test"].filter(lambda x: x != "fail").select(range(99))
            do_experiment(input_file, expected_output_file, mmlu_translated, politeness_levels)


