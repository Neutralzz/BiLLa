import sys, os, json

def main(question_file, reivew_result_file):
    score_a = {}
    score_b = {}
    # score_a = []
    # score_b = []
    questions = [None]
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
            assert questions[-1]["question_id"] + 1 == len(questions)
            score_a[questions[-1]["category"]] = []
            score_b[questions[-1]["category"]] = []
            
    with open(reivew_result_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            qid = sample["question_id"]
            ques = questions[qid]
            category = ques["category"]
            assert sample["score"][0] > 0 and sample["score"][1] > 0
            score_a[category].append(sample["score"][0])
            score_b[category].append(sample["score"][1])
    micro_score_a, micro_score_b = [], []
    macro_score_a, macro_score_b = [], []
    for category in score_a:
        N = len(score_a[category])
        assert len(score_b[category]) == N
        if N > 0:
            print(f"| {category} | {N} | {round(sum(score_a[category]) / N, 2)} | {round(sum(score_b[category]) / N, 2)}|")
            micro_score_a.extend(score_a[category])
            micro_score_b.extend(score_b[category])
            macro_score_a.append(sum(score_a[category]) / N)
            macro_score_b.append(sum(score_b[category]) / N)

    # print(len(micro_score_a), len(macro_score_a))
    print("Micro score:", sum(micro_score_a) / len(micro_score_a), sum(micro_score_b) / len(micro_score_b))
    print("Macro score:", sum(macro_score_a) / len(macro_score_a), sum(macro_score_b) / len(macro_score_b))


if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])
