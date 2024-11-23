import os

from openai import OpenAI

client = OpenAI(
    api_key="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",
)


def ask_gpt(formatted_question):
    response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": formatted_question
        }
    ],
    temperature = 0.2,
    model="gpt-4-turbo",

    )
    return response.choices[0].message.content


def parse_questions_file(file_path):
    data = []

    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file.readlines():
            if line.startswith("Subject: "):
                data.append([])
            elif line.startswith("politeness level: "):
                data[-1].append([])
            elif line.startswith("questions from "):
                data[-1][-1].append("")
            else:
                data[-1][-1][-1] = data[-1][-1][-1] + line
    return data


parsed_data = parse_questions_file("experiments/mmlu/heb/input/mmlu_input_heb.txt")

with open("mmlu_output_heb_4.txt", "w", encoding="utf-8") as output_file:
    for subject_data in parsed_data:
        output_file.write("Subject\n")
        for politeness_level_data in subject_data:
            output_file.write("politeness level\n")
            for question in politeness_level_data:
                try:
                    response = str(ask_gpt(question)) + "\n"
                    #response = "example" + "\n"
                    output_file.write(response)
                    output_file.flush()  # Flush data to OS buffer
                    os.fsync(output_file.fileno())
                except Exception as e:
                    print(e)
                    print("failed- continue")

print("finished")
