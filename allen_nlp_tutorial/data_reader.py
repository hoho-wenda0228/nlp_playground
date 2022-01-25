from typing import Dict, Iterable, List, Tuple

from allennlp.data import DatasetReader,Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer

class NestedNERTsvReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            max_tokens: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def _read(self, file_path:str) -> Iterable[Instance]:

