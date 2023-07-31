TAGS = dict(
    NUMERALS='numerals',
    NEGATION='negation',
    PASSIVE='passive',
    ROOT_VERB='root_verb',
    TRANSITIVITY='transitivity',
)

TAGS_SYMBOLS = dict(
    NUMERALS='NMR',
    NEGATION='NEG',
    PASSIVE='PAS',
    ROOT_VERB='RVB',
    TRANSITIVITY='TRN',
)

TAG_START = '<#'
TAG_END = '#> '


def create_tag(tag):
    return TAG_START + TAGS_SYMBOLS[tag] + TAG_END


def create_tokenizer(gpt2_type: str):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
    for tag in TAGS:
        tokenizer.add_special_tokens(
            {create_tag(tag): 'additional_special_tokens'}
        )
    return tokenizer


def load_tokenizer(path):
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
    return tokenizer
