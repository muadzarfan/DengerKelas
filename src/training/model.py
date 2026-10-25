"""One multilingual model/language contract for training, generation and reload."""
import copy
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from src.data.common import ROOT


def load_model(path=None):
    path = ROOT/'data/checkpoints/whisper-tiny-training' if path is None else path
    model = WhisperForConditionalGeneration.from_pretrained(str(path), local_files_only=True, attn_implementation='sdpa')
    if model.config.vocab_size != 51865 or model.config.decoder_start_token_id != 50258:
        raise ValueError('Expected multilingual Whisper Tiny')
    # Current Transformers uses explicit language/task, not legacy forced IDs.
    model.generation_config.language = 'id'
    model.generation_config.task = 'transcribe'
    model.generation_config.return_timestamps = False
    model.generation_config.max_length = 448
    model.generation_config.max_new_tokens = None
    model.generation_config.forced_decoder_ids = None
    if hasattr(model.config, 'forced_decoder_ids'):
        model.config.forced_decoder_ids = None
    model.config.use_cache = False  # no training KV cache; generation explicitly enables it
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def load_processor(path=None):
    path = ROOT/'data/checkpoints/whisper-tiny-processor' if path is None else path
    p = WhisperProcessor.from_pretrained(str(path), local_files_only=True, language='Indonesian', task='transcribe')
    p.tokenizer.set_prefix_tokens(language='Indonesian',task='transcribe',predict_timestamps=False)
    assert p.tokenizer.prefix_tokens == [50258,50275,50359,50363]
    return p


def generate(model, batch, max_new_tokens):
    # No reference/labels are passed to generation.
    config = copy.deepcopy(model.generation_config)
    # In Transformers 5.20 Whisper adds its 4 initial IDs to max_length inside
    # _set_max_new_tokens_and_length. Pass the content budget once, not twice.
    # This yields a total cap of 4 + budget (at most 448), using only one bound.
    config.max_length = min(model.config.max_target_positions - 4, max_new_tokens)
    config.max_new_tokens = None
    config.do_sample = False
    config.num_beams = 1
    config.use_cache = True
    config.return_dict_in_generate = True
    # Tensor-only Whisper output strips EOS. Raw sequences retain it so EOS/cap
    # checks are valid and checkpoint reload can compare exact decoder tokens.
    output = model.generate(input_features=batch['input_features'],attention_mask=batch['attention_mask'],
                            generation_config=config,force_unique_generate_call=True)
    return output.sequences
