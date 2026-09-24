import os
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def eval_single(annotation_file, result_file):
    annotations = json.load(open(annotation_file))
    annotations = {data["question_id"]: data for data in annotations}
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    for result in results:
        annotation = annotations[result["question_id"]]
        ground_truth = annotation["answer"]
        if "Unanswerable" in result["text"]:
            continue

        if ground_truth.lower() in result["text"].lower():
            right += 1

    print("Samples: {}\nAccuracy: {:.2f}%\n".format(total, 100.0 * right / total))

    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, "Result.text")
        with open(output_file, "w") as f:
            f.write(
                "Samples: {}\nAccuracy: {:.2f}%\n".format(total, 100.0 * right / total)
            )


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
