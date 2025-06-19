import re


def check_multiple_choice_with_regex(model_outputs, correct_answers):

    results = []
    for model_output, correct_answer in zip(model_outputs, correct_answers):
        correct_answer = correct_answer.upper()
        # Look for the answer letter at the beginning of a line or as the last
        patterns = [
            rf"\b{correct_answer}\b",  # Word boundary around the answer letter
            rf"\b{correct_answer}[.,)]",  # Answer followed by punctuation
            rf"\(.*{correct_answer}.*\)",  # Answer within parentheses
        ]

        match_found = False
        for pattern in patterns:
            if re.search(pattern, model_output):
                match_found = True
                break
        results.append(match_found)

    return results
