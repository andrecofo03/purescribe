from pathlib import Path
import argparse


def levenshtein(ref, hyp):
    n = len(ref)
    m = len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        ops[i][0] = "D"

    for j in range(1, m + 1):
        dp[0][j] = j
        ops[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1

            substitution = dp[i - 1][j - 1] + cost
            deletion = dp[i - 1][j] + 1
            insertion = dp[i][j - 1] + 1

            best = min(substitution, deletion, insertion)
            dp[i][j] = best

            if best == substitution:
                ops[i][j] = "C" if cost == 0 else "S"
            elif best == deletion:
                ops[i][j] = "D"
            else:
                ops[i][j] = "I"


    i = n
    j = m
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        op = ops[i][j]
        if op == "C":
            i -= 1
            j -= 1
        elif op == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif op == "D":
            deletions += 1
            i -= 1
        elif op == "I":
            insertions += 1
            j -= 1

    errors = substitutions + deletions + insertions
    return errors, substitutions, deletions, insertions


def load_reference(path):
    references = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip("\r\n")
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        filename = parts[0].strip()
        references[filename] = parts[1].strip().split() if len(parts) > 1 else []
    return references


def load_predictions(path):
    predictions = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip("\r\n")
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        filename = parts[0].strip()
        predictions[filename] = parts[1].strip().split() if len(parts) > 1 else []
    return predictions


def evaluate(reference_file, prediction_file, output_file="evaluation.txt"):
    references = load_reference(reference_file)
    predictions = load_predictions(prediction_file)

    total_words = 0
    total_errors = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0

    correct_sequences = 0
    total_sequences = len(references)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    for filename, ref in references.items():
        hyp = predictions.get(filename, [])
        errors, substitutions, deletions, insertions = levenshtein(ref, hyp)

        total_words += len(ref)
        total_errors += errors
        total_substitutions += substitutions
        total_deletions += deletions
        total_insertions += insertions

        if ref == hyp:
            correct_sequences += 1

        log(f"{filename}")
        log(f"  REF: {' '.join(ref)}")
        log(f"  HYP: {' '.join(hyp)}")
        log(f"  errors={errors} S={substitutions} D={deletions} I={insertions}")
        log()

    wer = total_errors / total_words if total_words > 0 else 0.0
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0


    log(f"Sequences:          {total_sequences}")
    log(f"Correct sequences:  {correct_sequences}")
    log(f"Sequence accuracy:  {sequence_accuracy * 100:.2f}%")
    log()
    log(f"Reference words:    {total_words}")
    log(f"Substitutions:      {total_substitutions}")
    log(f"Deletions:          {total_deletions}")
    log(f"Insertions:         {total_insertions}")
    log(f"Total errors:       {total_errors}")
    log(f"WER:                {wer * 100:.2f}%")

    out_path = Path(output_file)
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport di valutazione salvato in: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Calcola il WER e salva i risultati.")
    parser.add_argument("reference")
    parser.add_argument("predictions")
    parser.add_argument(
        "-o", "--output",
        default="evaluation.txt",
    )

    args = parser.parse_args()

    evaluate(
        args.reference,
        args.predictions,
        args.output
    )


if __name__ == "__main__":
    main()