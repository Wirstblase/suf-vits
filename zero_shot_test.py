import torch
import soundfile as sf
from speechbrain.pretrained import EncoderClassifier
import utils
import commons
from text import text_to_sequence
from models import SynthesizerTrn
from text.symbols import symbols


def get_text(text, hparams):
    text_normalized = text_to_sequence(text, hparams.data.text_cleaners)
    if hparams.data.add_blank:
        text_normalized = commons.intersperse(text_normalized, 0)
    text_normalized = torch.LongTensor(text_normalized)
    return text_normalized


def extract_speaker_embedding(audio_path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    signal, _ = sf.read(audio_path)
    embeddings = classifier.encode_batch(torch.tensor(signal).unsqueeze(0))
    return embeddings.squeeze().cpu().numpy()


# Load hyperparameters
hparams = utils.get_hparams_from_file("./configs/vctk_base.json")

# Initialize the synthesizer
synthesizer = SynthesizerTrn(
    len(symbols),
    hparams.data.filter_length // 2 + 1,
    hparams.train.segment_size // hparams.data.hop_length,
    n_speakers=hparams.data.n_speakers,
    **hparams.model).cuda()
synthesizer.eval()

# Load the pretrained model
utils.load_checkpoint("pretrained/pretrained_vctk.pth", synthesizer, None)

# Extract speaker embedding from a sample audio file of your voice
speaker_embedding = extract_speaker_embedding("samples/suflea_sample.wav")
speaker_embedding_tensor = torch.FloatTensor(speaker_embedding).unsqueeze(0).cuda()

# Prepare the input text
input_text = get_text("VITS is Awesome!", hparams)
with torch.no_grad():
    input_text_cuda = input_text.cuda().unsqueeze(0)
    input_text_lengths = torch.LongTensor([input_text.size(0)]).cuda()

    # Perform inference with the speaker embedding
    generated_audio = synthesizer.infer(
        input_text_cuda,
        input_text_lengths,
        sid=None,  # Not using the speaker ID since we have the speaker embedding
        spk_emb=speaker_embedding_tensor,
        noise_scale=.667,
        noise_scale_w=0.8,
        length_scale=1
    )[0][0, 0].data.cpu().float().numpy()

# Save the audio to a file
output_file = "output_audio_zeroshoot.wav"
sf.write(output_file, generated_audio, hparams.data.sampling_rate)

# Display a message indicating where the file has been saved
print(f"Audio saved to {output_file}")
