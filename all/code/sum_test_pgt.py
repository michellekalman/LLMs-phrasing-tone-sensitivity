import os

from openai import OpenAI

client = OpenAI(
    api_key="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",
)

politeness_levels = 8

def ask_gpt(formatted_text):
    response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": formatted_text
        }
    ],
    temperature = 0.2,
    model="gpt-4-turbo",

    )
    return response.choices[0].message.content


def parse_questions_file(file_path):
    data = []

    with open(file_path, 'r', encoding="utf-8") as file:
        sanity_counter = 0
        for line in file.readlines():
            if line.startswith("politeness level: "):
                data.append([])
            elif line.startswith("article "):
                sanity_counter += 1
                data[-1].append("")
            else:
                data[-1][-1] = data[-1][-1] + line
        if sanity_counter != 100 * politeness_levels:
            print(f"sanity counter failed with value {sanity_counter} instead {100 * politeness_levels}")
    return data


parsed_data = parse_questions_file("experiments/sum/heb/input/sum_input_heb.txt")

with open("sum_output_4_heb.txt", "w", encoding="utf-8") as output_file:
    for politeness_level_data in parsed_data:
        output_file.write("politeness level\n")
        for article in politeness_level_data:
            try:
                response = "Response: \n" + str(ask_gpt(article)) + "\n"
                #response = "Response: \n" + "example" + "\n"
                output_file.write(response)
                output_file.flush()  # Flush data to OS buffer
                os.fsync(output_file.fileno())
            except:
                print(f"failed for {article}- continue")

print("finished")
