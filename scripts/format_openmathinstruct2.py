from datasets import load_dataset


def format_record(rec):
    # Build new answer with think tags and required blank line
    generated = rec["generated_solution"]
    expected = rec["expected_answer"]
    answer = f"<think>{generated}</think>\n\nFinal Answer:{expected}"
    category = rec["problem_source"]
    return {
        "question": rec["problem"],
        "answer": answer,
        "category": category,
    }


def main():
    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train")
    ds = ds.map(format_record, remove_columns=ds.column_names)
    # Example: save to disk
    ds.to_json("../dataset/openmathinstruct2_formatted.jsonl", lines=True, force_ascii=True)

    # quick sanity check
    print(ds[0])


if __name__ == "__main__":
    main()
