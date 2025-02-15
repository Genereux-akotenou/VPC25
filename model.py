#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################

import whisper
from TTS.api import TTS
from IPython.display import Audio

def anonymize(input_audio_path): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`, 
        which ensures compatibility with `soundfile.write()`.
    sr : int
        The sample rate of the processed audio.
    """

    # init
    speech_to_test_model = whisper.load_model("base") # large-v2, medium
    test_to_speech_model = TTS("tts_models/multilingual/multi-dataset/bark", gpu=True)

    # Read the source audio file
    #audio = Audio("/content/1272-128104-0000.wav")

    # Apply your anonymization algorithm
    # 01
    result = speech_to_test_model.transcribe(input_audio_path)
    transcribe = result["text"]
    # 02
    # test_to_speech_model.tts_to_file(transcribe, file_path="out.wav")
    audio_array = test_to_speech_model.tts(transcribe)

    # Output:
    audio = audio_array
    sr = 22050
    
    return audio, sr