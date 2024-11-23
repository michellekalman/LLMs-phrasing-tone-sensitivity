from datasets import load_dataset
import random


"""
Create the dataset
"""


def not_too_long(data):
    if len(data['question']) > 66:
        return False
    for choice in data['choices']:
        if len(choice) > 66:
            return False
    return True


politeness_levels = 7


"""
Create input prompts
"""

question_prefix = ["", "You must ", "", "", "If you can, ", "If you don't mind, ", "If it's not too much trouble, "]
question_suffix = [" you scum bug!", ".", ".", ", please.", ", please.", ", please.", ", please."]
format_prefix = ["", "You must ", "", "", "", "", ""]
format_suffix = [" only!", "!", ".", ", please.", ", please.", ", please.", ", please."]


def create_question_prompts():
    prompts_list = []
    for politeness_level in range(politeness_levels):
        question_first_letter = "A" if question_prefix[politeness_level] == "" else "a"
        formatted_question = (question_prefix[politeness_level] + question_first_letter +
                              "nswer the following questions" + question_suffix[politeness_level] + "\n")
        prompts_list.append(formatted_question)
    return prompts_list


def create_format_prompts():
    prompts_list = []
    for politeness_level in range(politeness_levels):
        format_first_letter = "P" if format_prefix[politeness_level] == "" else "p"
        formatted_format = (format_prefix[politeness_level] + format_first_letter +
                            "rovide your answers in the following format" + format_suffix[politeness_level] + "\n1. A")
        prompts_list.append(formatted_format)
    return prompts_list


question_prompts = create_question_prompts()
format_prompts = create_format_prompts()


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
        letter_example = random.choice(["A", "B", "C", "D"])  # randomness is used in order to make it clear
        # that this is an example and prevent bias
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


with open("mmlu_input_eng.txt", "w", encoding="utf-8") as input_file:
    with open("mmlu_expected_output_eng.txt", "w", encoding="utf-8") as expected_output_file:
        for option in ["elementary_mathematics", "conceptual_physics", "philosophy", "high_school_macroeconomics", "all"]:
            input_file.write(f"Subject: {option}\n")
            expected_output_file.write(f"Subject: {option}\n")
            try:
                mmlu = load_dataset("cais/mmlu", option)
            except:
                print(f"can't load {mmlu}")
            l = len(mmlu["test"].filter(not_too_long))
            if l < 100:
                print(f"not enought question is subject {option}, {str(l)}")
            else:
                mmlu["test"] = mmlu["test"].filter(not_too_long).shuffle().select(range(99))
                do_experiment(input_file, expected_output_file, mmlu, politeness_levels)
