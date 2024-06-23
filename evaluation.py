import argparse
import evaluate


def evaluate_translations(
    teacher_predictions, student_predictions, references, sources=None
):
    # Evaluate with sacrebleu
    sacrebleu = evaluate.load("sacrebleu")

    teacher_sacrebleu_results = sacrebleu.compute(
        predictions=teacher_predictions, references=references
    )
    teacher_sacrebleu_score = round(teacher_sacrebleu_results["score"], 1)

    student_sacrebleu_results = sacrebleu.compute(
        predictions=student_predictions, references=references
    )
    student_sacrebleu_score = round(student_sacrebleu_results["score"], 1)

    # Evaluate with comet (if sources are provided)

    if sources:
        comet = evaluate.load("comet")
        teacher_comet_results = comet.compute(
            predictions=teacher_predictions,
            references=references,
            sources=sources,
            progress_bar=True,
        )
        teacher_comet_results = round(teacher_comet_results, 2)
        student_comet_results = comet.compute(
            predictions=student_predictions,
            references=references,
            sources=sources,
            progress_bar=True,
        )
        student_comet_results = round(student_comet_results, 2)

    # Evaluate with meteor
    meteor = evaluate.load("meteor")
    teacher_meteor_results = meteor.compute(
        predictions=teacher_predictions, references=references
    )
    teacher_meteor_score = round(teacher_meteor_results["meteor"], 2)

    student_meteor_results = meteor.compute(
        predictions=student_predictions, references=references
    )
    student_meteor_score = round(student_meteor_results["meteor"], 2)

    # Print results
    print("Teacher SacreBLEU Score:", teacher_sacrebleu_score)
    print("Student SacreBLEU Score:", student_sacrebleu_score)

    print("Teacher COMET Score:", teacher_comet_results["mean_score"])
    print("Student COMET Score:", student_comet_results["mean_score"])

    print("Teacher METEOR Score:", teacher_meteor_score)
    print("Student METEOR Score:", student_meteor_score)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate translations with SacreBLEU, COMET, and METEOR."
    )
    parser.add_argument(
        "--teacher_predictions",
        type=str,
        required=True,
        help="File path to teacher predictions.",
    )
    parser.add_argument(
        "--student_predictions",
        type=str,
        required=True,
        help="File path to student predictions.",
    )
    parser.add_argument(
        "--references", type=str, required=True, help="File path to references."
    )
    parser.add_argument(
        "--sources", type=str, required=False, help="File path to sources (optional)."
    )

    args = parser.parse_args()

    # Read inputs from files
    with open(args.teacher_predictions, "r") as f:
        teacher_predictions = [line.strip() for line in f.readlines()]

    with open(args.student_predictions, "r") as f:
        student_predictions = [line.strip() for line in f.readlines()]

    with open(args.references, "r") as f:
        references = [line.strip() for line in f.readlines()]

    if args.sources:
        with open(args.sources, "r") as f:
            sources = [line.strip() for line in f.readlines()]
    else:
        sources = None

    # Evaluate translations
    evaluate_translations(teacher_predictions, student_predictions, references, sources)


if __name__ == "__main__":
    main()
