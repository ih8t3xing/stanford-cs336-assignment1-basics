if __name__ == "__main__":
    from cs336_basics.train_bpe import train_bpe
    import time
    input_path = "tests/fixtures/corpus.en"
    start_time = time.time()
    _, _ = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    # assert end_time - start_time < 1.5
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    