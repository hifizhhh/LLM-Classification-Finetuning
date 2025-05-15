class CFG:
    seed = 42
    preset = "deberta_v3_base_en"
    sequence_length = 384
    epochs = 10
    batch_size = 8
    scheduler = "cosine"
    label2name = {0: "winner_model_a", 1: "winner_model_b", 2: "winner_tie"}
    name2label = {v: k for k, v in label2name.items()}
    class_labels = list(label2name.keys())
    class_names = list(label2name.values())
