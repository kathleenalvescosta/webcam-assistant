import os
import csv

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))

def formatter(root_path, manifest_file, **kwargs):
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for cols in reader:
            if len(cols) != 3:
                continue
            wav_file, text, speaker_name = cols
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker_name": speaker_name,
                "root_path": root_path
            })
    return items
dataset_config = BaseDatasetConfig(
	meta_file_train="metadata.csv",
	path=os.path.join(
		"/home/u6/kathleencosta/independentstudy"
	)
)

audio_config = VitsAudioConfig(
    sample_rate=22050, 
    win_length=1024, 
    hop_length=256, 
    num_mels=80, 
    mel_fmin=0, 
    mel_fmax=None
)

config = VitsConfig(
    audio=audio_config,
    run_name="vitseng",
    batch_size=2,
    eval_batch_size=2,
    batch_group_size=2,
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=50,
    text_cleaner=None,
    use_phonemes=True,
    phonemizer="gruut",
    phoneme_language="en-us",
    compute_input_seq_cache=True,
    print_step=250,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    phoneme_cache_path="/home/u6/kathleencosta/independentstudy/phoneme_cache",
    test_sentences=[
        ["Linguists need data, keep your mouth open!"]
    ]
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=formatter
)

model = Vits(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
	TrainerArgs(),
	config,
	output_path,
	model=model,
	train_samples=train_samples,
	eval_samples=eval_samples,
)

trainer.fit()
