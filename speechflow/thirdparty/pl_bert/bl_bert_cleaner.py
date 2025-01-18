import logging

from speechflow.logging import trace

__all__ = ["PLBertTextCleaner"]

LOGGER = logging.getLogger("root")

# IPA Phonemizer: https://github.com/bootphon/phonemizer
_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
pl_bert_symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)


class PLBertTextCleaner:
    def __init__(self, dummy=None):
        self.symb_to_id = {symb: idx for idx, symb in enumerate(pl_bert_symbols)}

    def __call__(self, text):
        indexes = []
        for s in text:
            try:
                indexes.append(self.symb_to_id[s])
            except KeyError:
                LOGGER.warning(trace(self, message=f"symbol [{s}] not in alphabet!"))
                indexes.append(self.symb_to_id["U"])  # unknown token

        return indexes
