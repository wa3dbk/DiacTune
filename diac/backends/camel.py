from __future__ import annotations
import re
from diac.backends.base import DiacritizationBackend
from camel_tools.morphology.database import MorphologyDB
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize

_WHITESPACE_RE = re.compile(r'\s+|\S+')


class CAMeLBackend(DiacritizationBackend):
    def __init__(self, db_name: str = "calima-msa-r13"):
        self._disambig = MLEDisambiguator.pretrained(db_name)

    def infer(self, sentences: list[str]) -> list[str]:
        results = []
        for sentence in sentences:
            tokens = _WHITESPACE_RE.findall(sentence)
            out_tokens = []
            for tok in tokens:
                if not tok.strip():
                    out_tokens.append(tok)
                else:
                    subtoks = simple_word_tokenize(tok)
                    disambig = self._disambig.disambiguate(subtoks)
                    out_tokens.extend(
                        d.analyses[0].analysis.get("diac", d.word)
                        for d in disambig
                    )
            results.append("".join(out_tokens))
        return results
