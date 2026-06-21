import torch

from utils import MODEL_REGISTRY, ATTENTION_REGISTRY

from config import (
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    ATTN_DIM,
    DROPOUT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Weight-file lookup table.
# ----------------------------
WEIGHT_PATHS = {
    ("vanilla_lstm",       "luong_general"): "../models/Vanilla LSTM with Luong General.pth",
    ("vanilla_lstm",       "bahdanau"):      "../models/Vanilla LSTM with Bahdanau.pth",
    ("bidirectional_lstm", "bahdanau"):      "../models/BiLSTM with Bahdanau.pth",
    ("bidirectional_lstm", "luong_concat"):  "../models/BiLSTM with LuongConcat.pth",
}

# Cache so each (model, attention) combo is only loaded once per process
_loaded_models: dict = {}


def get_model(model_name: str, attention_name: str, embedding_matrix):
    """
    Return a cached, eval-mode model for the requested
    (model_name, attention_name) combination.

    Raises KeyError if the combination has no trained weights.
    """
    key = (model_name, attention_name)

    if key in _loaded_models:
        return _loaded_models[key]

    if key not in WEIGHT_PATHS:
        available = ", ".join(f"{m}/{a}" for m, a in WEIGHT_PATHS)
        raise KeyError(
            f"No trained weights for model='{model_name}', "
            f"attention='{attention_name}'. "
            f"Available combinations: {available}"
        )

    ModelClass     = MODEL_REGISTRY[model_name]
    AttentionClass = ATTENTION_REGISTRY[attention_name]  # may be None for "none"

    model = ModelClass(
        embedding_matrix=embedding_matrix,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        attention_class=AttentionClass,
        attn_dim=ATTN_DIM,
        dropout=DROPOUT,
    )

    model.load_state_dict(
        torch.load(WEIGHT_PATHS[key], map_location=device)
    )

    model.to(device)
    model.eval()

    _loaded_models[key] = model
    return model