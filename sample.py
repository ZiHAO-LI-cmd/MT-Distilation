import random
import argparse


def read_sentences(file_path):
    """Read sentences from a file, stripping trailing characters."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file]


def write_sentences(sentences, file_path):
    """Write sentences to a file."""
    with open(file_path, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")


def sample_parallel_sentences(
    src_file_path,
    tgt_file_path,
    output_src_path,
    output_tgt_path,
    sample_size,
    seed
):
    random.seed(seed)
    try:
        src_sentences = read_sentences(src_file_path)
        tgt_sentences = read_sentences(tgt_file_path)

        assert len(src_sentences) == len(
            tgt_sentences
        ), "The number of sentences must be the same in both files."

        indices = (
            random.sample(range(len(src_sentences)), sample_size)
            if sample_size is not None
            else range(len(src_sentences))
        )
        sampled_src_sentences = [src_sentences[i] for i in indices]
        sampled_tgt_sentences = [tgt_sentences[i] for i in indices]

        write_sentences(sampled_src_sentences, output_src_path)
        write_sentences(sampled_tgt_sentences, output_tgt_path)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample parallel sentences from source and target text files."
    )
    parser.add_argument("src_file_path", help="Path to the source text file.")
    parser.add_argument("tgt_file_path", help="Path to the target text file.")
    parser.add_argument(
        "output_src_path", help="Path to output the sampled source sentences."
    )
    parser.add_argument(
        "output_tgt_path", help="Path to output the sampled target sentences."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of sentences to sample. If not specified, all sentences are used.",
    )
    parser.add_argument(
        "--seed", type=int, default=666, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    sample_parallel_sentences(
        args.src_file_path,
        args.tgt_file_path,
        args.output_src_path,
        args.output_tgt_path,
        args.sample_size,
        args.seed,
    )

# sample_parallel_sentences("./data/en-uk/NLLB.en-uk.en", "./data/en-uk/NLLB.en-uk.uk", "./data/en-uk/sampled.en", "./data/en-uk/sampled.uk", sample_size=100)
