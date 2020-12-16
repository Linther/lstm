import random
import re
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers , activations , models , preprocessing , utils
from nltk.translate.bleu_score import sentence_bleu


class cornellConvo:
    def __init__(self, convoId):
        self.convoId = convoId
    def addLineToConvo(self, sentence):
        self.sentenceList.append()

class cornellLine:
    def __init__(self, lineId, lineString, characterId):
        self.lineId = lineId
        self.lineString = lineString
        self.characterId = characterId
    def append(self, additionalLine):
        self.lineString.append(additionalLine)
# # I only want conversations that have an even number of messages with an alternating speaker. e.g. Sarah, Jared, Sarah, Jared, Sarah, Jared...
# def validateConvo():



def main():
    newModelMode = False
    bleuMode = False
    batchSize = 32
    epochCount = 1
    modelPath = ""
    movieConvoPath = ""
    for i, commandLineArg in enumerate(sys.argv):
        if (commandLineArg == "-load") and (len(sys.argv) - i >= 2):
            modelPath = sys.argv[i + 1]
            movieConvoPath = sys.argv[i + 2]
        elif (commandLineArg == "-train"):
            modelPath = sys.argv[i + 1]
            movieConvoPath = sys.argv[i + 2]
            batchSize = int(sys.argv[i + 3])
            epochCount = int(sys.argv[i + 4])
            newModelMode = True
        elif commandLineArg == "-bleu":
            bleuMode = True
        else:
            print("Incorrect Usage! Please consult the README for instructions")
    
    if not movieConvoPath:
        movieConvoPath = "cornell_corpus\\72movies.txt"
    # print (modelPath)



    movieLinesreader = open("Datasets\cornell movie-dialogs corpus\movie_lines.txt")
    previousLineId = 0
    previousLineCharacter = 0
    linesDic = {}
    for line in movieLinesreader.readlines():
        splitLine = line.lower().split(" +++$+++ ")
        if (splitLine[1] == previousLineCharacter) and (int(previousLineId[1:]) - int(splitLine[0][1:]) == 0):
            linesDic[previousLineId].lineString = re.sub("\n", "", (splitLine[4] + linesDic[previousLineId].lineString))
        else:
            linesDic[splitLine[0]] = cornellLine(splitLine[0],  re.sub("\n", "",splitLine[4]), splitLine[1])
            previousLineCharacter = splitLine[1]
            previousLineId = splitLine[0]


    inputLines = []
    outputLines = []
    inputFlag = 0
    fileReader = open(movieConvoPath)
    for line in fileReader.readlines():
        splitLine = line.split(" +++$+++ ")
        convoPointerList = splitLine[3]
        convoPointerList = re.sub(r"(\[\'|\'\]|\n)", "", convoPointerList.lower()) #removing initial/trailing brackets of ['L204', 'L205', 'L206'] convos
        convoPointerList = convoPointerList.split("\', \'")
        for lineId in convoPointerList:
            if not linesDic.get(lineId):
                continue
            elif inputFlag == 0:
                inputLines.append(linesDic[lineId].lineString)
                inputFlag = 1
            elif inputFlag == 1:
                outputLines.append(linesDic[lineId].lineString)
                inputFlag = 0


    if (len(inputLines) != len(outputLines)):
        del outputLines[len(inputLines):]
        del inputLines[len(outputLines):]


    inputTokenizer = preprocessing.text.Tokenizer()

    # train tokenizer on input data
    inputTokenizer.fit_on_texts(inputLines)

    # vectorize the input lines
    tokenizedInputLines = inputTokenizer.texts_to_sequences(inputLines)

    maxInputSeqLength = len(max(tokenizedInputLines, key=len))

    # reshaping array sizes for the encoder
    paddedInputSeqLines = preprocessing.sequence.pad_sequences(tokenizedInputLines , maxlen=maxInputSeqLength , padding='post')
    encoderInputData = np.array(paddedInputSeqLines)

    #determining vocabulary size
    inputVocab = inputTokenizer.word_index
    inputVocabSize = len(inputVocab) + 1




    # decoder stuff

    decoderLines = []
    for line in outputLines:
        decoderLines.append("<START> " + line + " <END>")

    decoderTokenizer = preprocessing.text.Tokenizer()

    # train tokenizer on decoder data
    decoderTokenizer.fit_on_texts(decoderLines)

    # vectorize the decoder lines
    tokenizedDecoderLines = decoderTokenizer.texts_to_sequences(decoderLines)


    maxDecoderSeqLength = len(max(tokenizedDecoderLines, key=len))

    paddedDecoderSeqLines = preprocessing.sequence.pad_sequences(tokenizedDecoderLines, maxlen=maxDecoderSeqLength, padding='post')
    decoderInputData = np.array(paddedDecoderSeqLines)

    decoderVocab = decoderTokenizer.word_index
    decoderVocabSize = len(decoderVocab) + 1


    # Decoder Target Data


    decoderTargetData = []
    if newModelMode == True:
        print("Creating one-hot target vector. This might take a while please be patient.")

        for seq in tokenizedDecoderLines:
            decoderTargetData.append(seq[1:])

        paddedDecoderTargetData = preprocessing.sequence.pad_sequences(decoderTargetData, maxlen=maxDecoderSeqLength, padding = 'post')
        onehotTargetData = utils.to_categorical(paddedDecoderTargetData, decoderVocabSize)
        decoderTargetData = np.array(onehotTargetData)




    # Training model

    # encoder

    modelEncoderInput = tf.keras.layers.Input(shape=(None,))
    modelEncoderEmbedding = tf.keras.layers.Embedding(inputVocabSize, 256, mask_zero = True)(modelEncoderInput)
    modelEncoderOutputs, state_h, state_c = tf.keras.layers.LSTM(128, return_state=True)(modelEncoderEmbedding)
    modelEncoderStates = [state_h, state_c]


    # decoder

    modelDecoderInput = tf.keras.layers.Input(shape=(None,))
    modelDecoderEmbedding = tf.keras.layers.Embedding(decoderVocabSize, 256, mask_zero = True)(modelDecoderInput)
    modelDecoderLSTM = tf.keras.layers.LSTM(128, return_state= True, return_sequences = True)
    modelDecoderOutputs, _ , _ = modelDecoderLSTM(modelDecoderEmbedding, initial_state=modelEncoderStates)
    modelDecoderDense = tf.keras.layers.Dense(decoderVocabSize, activation=tf.keras.activations.softmax)
    output = modelDecoderDense(modelDecoderOutputs)

    # model

    model = tf.keras.models.Model([modelEncoderInput, modelDecoderInput], output)


    if (newModelMode == True):
        print ("Compiling Model...")
        model.compile(optimizer = tf.keras.optimizers.RMSprop(), loss = 'categorical_crossentropy')
        model.fit([encoderInputData, decoderInputData], decoderTargetData, batch_size=batchSize, epochs= epochCount,shuffle=True) 
        model.save(modelPath)

    if newModelMode == False:
        model = tf.keras.models.load_model(modelPath)
    
    model.summary()


    def make_inference_models():
        
        encoder_inputs = model.input[0]
        encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output
        encoder_states = [state_h_enc, state_c_enc]
        decoder_inputs = model.input[1]
        decoder_embedding = model.layers[3].output
        encoderModel = tf.keras.models.Model(encoder_inputs, encoder_states)
        
        decoder_state_input_h = tf.keras.layers.Input(shape=( 128 ,))
        decoder_state_input_c = tf.keras.layers.Input(shape=( 128 ,))
        
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[5]
        decoder_dense = model.layers[6]

        decoder_outputs, state_h, state_c = decoder_lstm( decoder_embedding , initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoderModel = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        
        return encoderModel , decoderModel

    def str_to_tokens( sentence : str ):
        words = sentence.lower().split()
        tokens_list = list()
        for word in words:
            if not inputVocab.get(word):
                tokens_list.append( random.choice(list(inputVocab.values())))
            else:
                tokens_list.append( inputVocab[ word ] ) 
        return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxInputSeqLength , padding='post')

    enc_model , dec_model = make_inference_models()


    quitCharacter = 'q'
    userInput = 'a'
    randomLineIndex = 0
    while userInput != quitCharacter:
        if bleuMode == False:
            userInput = input( 'Chat: ' )
        else:
            input( 'Any Key for next line: ')
            randomLineIndex = random.randrange(len(tokenizedInputLines))
            userInput = inputLines[randomLineIndex]
        states_values = enc_model.predict(str_to_tokens(userInput))
        #states_values = enc_model.predict( encoder_input_data[ epoch ] )
        empty_target_seq = np.zeros( ( 1 , 1 ) )
        empty_target_seq[0, 0] = decoderVocab['start']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition :
            dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
            sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
            sampled_word = None
            for word , index in decoderVocab.items() :
                if sampled_word_index == index :
                    sampled_word = word
                    if sampled_word == 'end' or len(decoded_translation.split()) > maxDecoderSeqLength:
                        stop_condition = True
                        break
                    decoded_translation += ' {}'.format( word )
                
            empty_target_seq = np.zeros( ( 1 , 1 ) )  
            empty_target_seq[ 0 , 0 ] = sampled_word_index
            states_values = [ h , c ] 

        print( decoded_translation )
        if bleuMode == True:
            print(outputLines[randomLineIndex].split(), "|", decoded_translation.split())
            print("Bleu score vs corpus: ", sentence_bleu(outputLines[randomLineIndex].split(), decoded_translation.split()))

    print ("Model Shutdown, Thank you!")


if __name__ == "__main__":
    main()