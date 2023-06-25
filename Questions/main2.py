from semanticParser2 import *
from torchtext.data import BucketIterator

if __name__ == "__main__":
    preproc = Preprocessor('E:/TFM/Dataset1Car/datasetQuestions/questionsDatasets_Small_OR.csv')

    # Get the dataset object
    train_data = preproc.train_data
    val_data = preproc.val_data

    vocF_prog_f=preproc.prog_f.vocab.stoi
    vocF_que_f = preproc.que_f.vocab.stoi

    # Looking at the Vocabulary
    print(preproc.prog_f.vocab.stoi)

    # Looking at the Vocabulary
    print(preproc.que_f.vocab.stoi)

    #Training
    # Training hyperparameters
    num_epochs = 5
    learning_rate = 3e-4
    batch_size = 150
    num_steps = len(train_data) / batch_size

    # Model hyperparameters
    config = {
        'que_vocab_size': len(preproc.que_f.vocab),
        'prog_vocab_size': len(preproc.prog_f.vocab),
        'embedding_dim': 256*2,
        'num_heads': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dropout': 0.10,
        'max_len': 800,
        'forward_expansion': 4,
        'que_pad_idx': preproc.que_f.vocab.stoi["<pad>"]
    }

    config['max_len']=800
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Training Generator
    train_loader1 = BucketIterator.splits((train_data,),
                                         batch_size=batch_size,
                                         sort_within_batch=True,
                                         sort_key=lambda x: len(x.query),
                                         device=device)[0]

    val_loader1 = BucketIterator.splits((val_data,),
                                          batch_size=batch_size,
                                          sort_within_batch=True,
                                          sort_key=lambda x: len(x.query),
                                          device=device)[0]

    # Create Model
    seq2seq = Seq2Seq(config)

    # Train Model
    seq2seq.train_model(train_loader1, val_loader1, num_epochs, num_steps, filename='semantic_parser.pth')

    sem_parser = SemanticParser(preproc, config)

    program = sem_parser.predict('Are any acura  cl cars visible?')

    print(program)
    # program = sem_parser.predict('What is in the image?')
    # print(program)
    # program = sem_parser.predict('What manufacter is the car?')
    # print(program)
    # program = sem_parser.predict('What company is the car?')
    # print(program)
    # # program = sem_parser.predict('What is the color of the car?')
    # # print(program)
    # program = sem_parser.predict('What is next to the car?')
    # print(program)
    #
