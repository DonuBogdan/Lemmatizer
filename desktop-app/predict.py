import numpy as np
from keras.models import load_model
from keras import backend as K
K.clear_session()


MAX_LEN_INPUT = 50
MAX_LEN_OUTPUT = 52
LIST_INPUT_CHARACTERS_MODEL_1 = [' ', '%', '&', "'", ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '@', 'A', 
'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 
'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ª', 'µ', 'º', 'À', 'Á', 'Â', 'Å', 
'Ç', 'È', 'É', 'Ë', 'Î', 'Ó', 'Ö', 'Ø', 'Þ', 'ß', 'à', 'á', 'â', 'ä', 'å', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ñ', 'ò', 'ó', 'ô', 'õ', 
'ö', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'Ă', 'ă', 'ć', 'č', 'ė', 'ğ', 'İ', 'ı', 'ł', 'ō', 'Ő', 'ő', 'œ', 'ŕ', 'Š', 'š', 'ū', 'ű', 'ž', 'Ș', 'ș', 'Ț', 
'ț', '˚', '˝', 'Δ', 'α', 'β', 'γ', 'δ', 'ε', 'η', 'θ', 'κ', 'λ', 'μ', 'ν', 'π', 'ρ', 'σ', 'τ', 'ω', 'Ф', '′', '″', 'Ω', '→', '∞', '≈']
LIST_OUTPUT_CHARACTERS_MODEL_1 = ['\t', '\n', ' ', '%', '&', "'", ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', 
';', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 
'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~', 'ª', 'µ', 'º', 
'À', 'Á', 'Â', 'Å', 'Ç', 'È', 'É', 'Ë', 'Î', 'Ó', 'Ö', 'Ø', 'Û', 'Þ', 'ß', 'à', 'á', 'â', 'ä', 'å', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 
'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'Ă', 'ă', 'ć', 'č', 'ė', 'ğ', 'İ', 'ı', 'ł', 'ō', 'Ő', 'ő', 'œ', 'ŕ', 'Š', 'š', 'ū', 
'ű', 'ž', 'Ș', 'ș', 'Ț', 'ț', '˚', '˝', 'Δ', 'α', 'β', 'γ', 'δ', 'ε', 'η', 'θ', 'κ', 'λ', 'μ', 'ν', 'π', 'ρ', 'σ', 'τ', 'ω', 'Ф', '′', '″', 'Ω', 
'→', '∞', '≈']
encoder_first_model = load_model('s2s_first_model_encoder.h5')
decoder_first_model = load_model('s2s_first_model_decoder.h5')


def necessary_dictionaries(LIST_INPUT_CHARACTERS, LIST_OUTPUT_CHARACTERS):

    input_token_index = dict([(char, i) for i, char in enumerate(LIST_INPUT_CHARACTERS)])
    target_token_index = dict([(char, i) for i, char in enumerate(LIST_OUTPUT_CHARACTERS)])

    # Reverse-lookup token index to decode sequences back to something readable
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    return(input_token_index, target_token_index, reverse_input_char_index, reverse_target_char_index)

def get_encoder_input_data(list_of_words, LIST_INPUT_CHARACTERS, MAX_LEN_INPUT):

    encoder_input_data = np.zeros((len(list_of_words), MAX_LEN_INPUT, len(LIST_INPUT_CHARACTERS)), dtype = 'float32')
    
    for i, input_text in enumerate(list_of_words):
        
        for index, char in enumerate(input_text):
            encoder_input_data[i, index, input_token_index[char]] = 1.    
            
        encoder_input_data[i, index + 1:, input_token_index[' ']] = 1.
    
    return(encoder_input_data)

def decode_sequence_first_model(input_seq, LIST_OUTPUT_CHARACTERS):
    # Encode the input as state vectors
    states_value = encoder_first_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, len(LIST_OUTPUT_CHARACTERS)))
    
    # Populate the first character of target sequence with the start character
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1)
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_first_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character
        if (sampled_char == '\n' or len(decoded_sentence) > MAX_LEN_OUTPUT):
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1, len(LIST_OUTPUT_CHARACTERS)))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def predict_func(list_of_words):

    global input_token_index
    global target_token_index
    global reverse_input_char_index
    global reverse_target_char_index
    
    list_of_predictions = []

    # Prepare the data for the first model
    input_token_index, target_token_index, reverse_input_char_index, reverse_target_char_index = necessary_dictionaries(LIST_INPUT_CHARACTERS_MODEL_1, LIST_OUTPUT_CHARACTERS_MODEL_1)
    encoder_input_data = get_encoder_input_data(list_of_words, LIST_INPUT_CHARACTERS_MODEL_1, MAX_LEN_INPUT)

    for seq_index in range(len(list_of_words)):

        input_seq = encoder_input_data[seq_index: seq_index + 1]

        decoded_sentence_first_model = decode_sequence_first_model(input_seq, LIST_OUTPUT_CHARACTERS_MODEL_1)
    
        list_of_predictions.append(decoded_sentence_first_model)

    return list_of_predictions