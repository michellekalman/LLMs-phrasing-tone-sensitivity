import matplotlib.pyplot as plt
import numpy as np

choice_index = {"A":"0", "B":"1", "C":"2", "D":"3"}


def compare_by_batch(response, expected):
    try:
        if str([int(choice_index[answer[3]]) for answer in response]) == expected[:-1]:
            return 1
        else:
            return 0
    except:
        return 0


def compare_by_singles(response, expected):
    result = 0
    for i in range(3):
        try:
            if [(choice_index[answer[3]]) for answer in response][i] == expected[:-1][3*i+1]:
                result += 1
        except:
            result += 0
    return result


def compare_all(compare_func):
    with open("experiments/mmlu/heb/output/4/mmlu_output_heb_4_fixed.txt", "r", encoding="utf-8") as output_file:
        with open("experiments/mmlu/heb/expected_output/mmlu_expected_output_heb.txt", "r", encoding="utf-8") as expected_output_file:
            output_file_lines = output_file.readlines()
            expected_output_file_lines = expected_output_file.readlines()
            expected_output_file_idx = 0
            results = []
            output_line_idx = 0
            while output_line_idx < len(output_file_lines):
                if output_file_lines[output_line_idx].startswith("Subject"):
                    results.append([])
                    output_line_idx += 1
                    expected_output_file_idx += 1
                elif output_file_lines[output_line_idx].startswith("politeness level"):
                    results[-1].append([])
                    output_line_idx += 1
                    expected_output_file_idx += 1
                else:
                    expected_output_file_idx += 1
                    #results[-1][-1].append(f"[{output_file_lines[output_line_idx]}, {output_file_lines[output_line_idx+1]}, {output_file_lines[output_line_idx+2]}]")
                    results[-1][-1].append(compare_func([output_file_lines[output_line_idx], output_file_lines[output_line_idx + 1], output_file_lines[output_line_idx + 2]], expected_output_file_lines[expected_output_file_idx]))
                    output_line_idx += 3
                    expected_output_file_idx += 1
    return results


results_by_batch = compare_all(compare_by_batch)

results_by_singles = compare_all(compare_by_singles)

for subject_data in results_by_batch:
    print([sum(p_level_data)/(len(p_level_data)*3)*100 for p_level_data in subject_data])

print("\n")
for subject_data in results_by_singles:
    print([sum(p_level_data)/(len(p_level_data)*3)*100 for p_level_data in subject_data])

subjects = ["elementary_mathematics", "conceptual_physics", "philosophy", "high_school_macroeconomics", "all"]

for sub_idx, subject_data in enumerate(results_by_batch):
    plt.plot([sum(p_level_data)/(len(p_level_data)*3)*100 for p_level_data in subject_data], label=f"Subject {subjects[sub_idx]}")

plt.xlabel("Politeness")
plt.ylabel("Accuracy of single questions")
plt.title("Line Plot of Accuracy of batch questions Over Politeness")
plt.legend()
plt.show()

print("finished")
