""" ABCTokenizer for Hugging Face Transformers. 

Is a character tokenizer that uses all ascii characters, digits, punctuation and space.
"""
import json
import os
import string
from pathlib import Path
from typing import Dict, List, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

class ABCTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        """ABCTokenizer for Hugging Face transformers."""
        self.characters = list(string.ascii_letters) + list(string.digits) + list(string.punctuation) + ['\n', ' ']
        
        self.pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        self.unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        
        self._vocab_str_to_int = {
            "[PAD]": 0,
            "[UNK]": 1,
            **{ch: i + 2 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            add_prefix_space=False,
            **kwargs,
        )


    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    @classmethod
    def from_config(cls, config: Dict) -> "ABCTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)
