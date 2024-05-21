import random
import argparse

def stream_write_sampled_sentences(src_file_path, tgt_file_path, output_src_path, output_tgt_path, sample_size, seed):
    random.seed(seed)
    try:
        # Determine the total number of lines in one of the files
        with open(src_file_path, "r", encoding="utf-8") as file:
            total_lines = sum(1 for line in file)
        
        # If sample_size is None or larger than the file, use all lines
        if sample_size is None or sample_size >= total_lines:
            indices = set(range(total_lines))
        else:
            indices = set(random.sample(range(total_lines), sample_size))
        
        # Stream through both files and write sampled lines
        with open(src_file_path, "r", encoding="utf-8") as src_file, \
             open(tgt_file_path, "r", encoding="utf-8") as tgt_file, \
             open(output_src_path, "w", encoding="utf-8") as out_src_file, \
             open(output_tgt_path, "w", encoding="utf-8") as out_tgt_file:
            for i, (src_line, tgt_line) in enumerate(zip(src_file, tgt_file)):
                if i in indices:
                    out_src_file.write(src_line)
                    out_tgt_file.write(tgt_line)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample parallel sentences from source and target text files efficiently."
    )
    parser.add_argument("src_file_path", help="Path to the source text file.")
    parser.add_argument("tgt_file_path", help="Path to the target text file.")
    parser.add_argument("output_src_path", help="Path to output the sampled source sentences.")
    parser.add_argument("output_tgt_path", help="Path to output the sampled target sentences.")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of sentences to sample. If not specified, all sentences are used.")
    parser.add_argument("--seed", type=int, default=666, help="Random seed for reproducibility.")

    args = parser.parse_args()
    stream_write_sampled_sentences(
        args.src_file_path,
        args.tgt_file_path,
        args.output_src_path,
        args.output_tgt_path,
        args.sample_size,
        args.seed,
    )
