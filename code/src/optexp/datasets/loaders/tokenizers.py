import os
import re
from torchtext.data.functional import generate_sp_model
from torchtext.transforms import SentencePieceTokenizer

_patterns = [
    r"\'",
    r"\"",
    r"\.",
    r"<br \/>",
    r",",
    r"\(",
    r"\)",
    r"\!",
    r"\?",
    r"\;",
    r"\:",
    r"\s+",
]

_replacements = [
    " '  ",
    "",
    " . ",
    " ",
    " , ",
    " ( ",
    " ) ",
    " ! ",
    " ? ",
    " ",
    " ",
    " ",
]

_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))


def _char_level_tokenizer(line):
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return list(line)


def _get_bpe_tokenizer(train_file, tokenizer_save_path):
    path_to_bpe_model = tokenizer_save_path / f"{str(train_file.stem)}.model"
    if not os.path.isfile(path_to_bpe_model):
        raise ValueError(
            """Tokenizer file is required. Call the _create_bpe_tokenizer function in this file
            using the same train_file argument and a vocab size. You will find a .model file 
            in your current working directory after the _create_bpe_tokenizer has finished running. \
            Create a folder named tokenizers in the optexp workspace and put the .model file in that folder.
            Remove the call to _create_bpe_tokenizer and resume. 
            
        """
        )
    return SentencePieceTokenizer(str(path_to_bpe_model))


def _create_bpe_tokenizer(train_file, vocab_size):
    generate_sp_model(
        str(train_file),
        model_type="bpe",
        vocab_size=vocab_size,
        model_prefix=str(train_file.stem),
    )
