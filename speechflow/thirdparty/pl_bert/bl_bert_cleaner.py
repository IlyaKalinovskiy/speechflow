# IPA Phonemizer: https://github.com/bootphon/phonemizer

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
pl_bert_symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

__all__ = ["PLBertTextCleaner"]


class PLBertTextCleaner:
    def __init__(self, dummy=None):
        self.symb_to_id = {symb: idx for idx, symb in enumerate(pl_bert_symbols)}

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.symb_to_id[char])
            except KeyError:
                indexes.append(self.symb_to_id["U"])  # unknown token
        return indexes
