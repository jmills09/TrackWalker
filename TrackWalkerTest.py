import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from MiscFunctions import *
from DataLoader import *

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_steps(seq,long=False):
    full_np = np.stack(seq,axis=0)
    if not long:
        return torch.tensor(full_np, dtype=torch.float)
    else:
        return torch.tensor(full_np, dtype=torch.long)

HIDDEN_DIM = 1024
PADDING = 30
EMBEDDING_DIM = (PADDING*2+1)*(PADDING*2+1) # N_Features
NUM_CLASSES = (PADDING*2+1)*(PADDING*2+1)+1 # Bonus Class is for the end of track class
EPOCHS = 1000
INFILE = "/home/jmills/workdir/TrackWalker/inputfiles/merged_dlreco_75e9707a-a05b-4cb7-a246-bedc2982ff7e.root"
REAL_IM = True
N_STEPS =-1 # -1 -> all steps, otherwise only first n_steps
TRACK_IDX = 0
EVENT_IDX = 0

def main():
     # Hyper Parameters:

    # return 0



    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    torch.manual_seed(1)

    input_image_dimension = PADDING*2+1
    steps_x = []
    steps_y = []
    full_image = []
    print()
    print()
    print("Real Image?", REAL_IM)
    print()
    print()
    if not REAL_IM:
        steps_x = [0,1,2,3,5,6,8,10,11,14,16,17,19,21,22,23,26,28,30,32,33,34,35,39,41,42,44,49]
        steps_y = [0,1,2,3,5,6,8,10,11,14,16,17,19,21,22,23,26,28,30,32,33,34,35,39,41,42,44,49]
        full_image = dumb_diag_test(steps_x)
    else:
        image_list,xs,ys = load_rootfile(INFILE)
        save_im(image_list[EVENT_IDX],"images/EventDisp")
        full_image = image_list[EVENT_IDX]
        steps_x = xs[EVENT_IDX][TRACK_IDX]
        steps_y = ys[EVENT_IDX][TRACK_IDX]

    if N_STEPS !=-1:
        steps_x = steps_x[0:N_STEPS]
        steps_y = steps_y[0:N_STEPS]
    # return 0
    print(len(steps_x), " Track Points")
    print(steps_x)
    print(steps_y)
    print('dx and dy')
    for ix in range(len(steps_x)-1):
        print(steps_x[ix+1]-steps_x[ix],end=' ')
    print()
    for ix in range(len(steps_y)-1):
        print(steps_y[ix+1]-steps_y[ix],end=' ')
    print()
    print()

    # save_im(full_image,'file')
    stepped_images = []
    flat_stepped_images = []
    next_positions = []
    flat_next_positions = []

    for idx in range(len(steps_x)):
        step_x = steps_x[idx]
        step_y = steps_y[idx]
        next_step_x = -1.0
        next_step_y = -1.0
        if idx != len(steps_x)-1:
            next_step_x = steps_x[idx+1]
            next_step_y = steps_y[idx+1]
        cropped_step_image = cropped_np(full_image, step_x, step_y, PADDING)
        required_padding_x = PADDING - step_x
        required_padding_y = PADDING - step_y
        stepped_images.append(cropped_step_image)
        flat_stepped_images.append(unravel_array(cropped_step_image))
        flat_next_positions_array = np.zeros(input_image_dimension*input_image_dimension+1)
        if idx != len(steps_x)-1:
            target_x = required_padding_x + next_step_x
            target_y = required_padding_y + next_step_y
            np_step_target = np.array([target_x*1.0,target_y*1.0])
            flat_np_step_target = target_x*cropped_step_image.shape[1]+target_y
            next_positions.append(np_step_target)
            flat_next_positions.append(flat_np_step_target)
            # flat_next_positions_array[flat_np_step_target] = 1
            # flat_next_positions.append(flat_next_positions_array)
        else:
            next_positions.append(np.array([-1.0,-1.0]))
            # flat_next_positions_array[flat_next_positions_array.shape[0]-1] = 1
            # flat_next_positions.append(flat_next_positions_array)
            flat_next_positions.append(NUM_CLASSES-1)

    print("flat_next_positions")
    print(flat_next_positions)

        # save_im(cropped_step_image,'step'+str(idx))
        # print(next_positions[len(next_positions)-1])

    # print(stepped_images)
    # print(next_positions)

    # Step 1: Make Full Image  -- Done
    # Step 2: Make List of Stepped Images
    # Step 3: Make List of Stepped Images unraveled/flattened
    # Step 4: Make matching list of next step coordinates
    # Step 5: Feed into Network

    ######################################################
    ######################################################
    ######################################################
    # lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
    # inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
    #
    # # initialize the hidden state.
    # hidden = (torch.randn(1, 1, 3),
    #           torch.randn(1, 1, 3))
    # for i in inputs:
    #     # Step through the sequence one element at a time.
    #     # after each step, hidden contains the hidden state.
    #     out, hidden = lstm(i.view(1, 1, -1), hidden)
    #
    # # alternatively, we can do the entire sequence all at once.
    # # the first value returned by LSTM is all of the hidden states throughout
    # # the sequence. the second is just the most recent hidden state
    # # (compare the last slice of "out" with "hidden" below, they are the same)
    # # The reason for this is that:
    # # "out" will give you access to all hidden states in the sequence
    # # "hidden" will allow you to continue the sequence and backpropagate,
    # # by passing it as an argument  to the lstm at a later time
    # # Add the extra 2nd dimension
    # inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    # hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
    # out, hidden = lstm(inputs, hidden)
    # print(out)
    # print(hidden)
    # ######################################################
    # ######################################################
    # ######################################################



    print(len(flat_next_positions), "len(flat_next_positions)")
    training_data = [
        # Tags are: DET - determiner; NN - noun; V - verb
        # For example, the word "The" is a determiner
        (flat_stepped_images,flat_next_positions)
        # ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        # ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]

    ix_to_word = {}
    # For each words-list (sentence) and tags-list in each tuple of training_data
    for sent, tags in training_data:
        for ix,flat_stepped_image in enumerate(sent):
            is_in = False
            for iii in range(len(ix_to_word)):
                if np.array_equal(flat_stepped_image, ix_to_word[iii]):
                    print("--------------------------------------------")
                    print("Is already in.")
                    print("--------------------------------------------")
                    is_in = True
                    break
            if is_in == False:
                ix_to_word[len(ix_to_word)] = flat_stepped_image  # Assign each word with a unique index



    # tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index
    # Label the potential targets with unique indices
    tag_to_ix = {}
    tag_to_ix[-1] = -1 #The End of Sequence Label
    for xco in range(input_image_dimension):
        for yco in range(input_image_dimension):
            tag_to_ix[xco*input_image_dimension+yco] = xco*input_image_dimension+yco



    ######################################################
    ######################################################
    ######################################################

    class LSTMTagger(nn.Module):

        def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            super(LSTMTagger, self).__init__()
            self.hidden_dim = hidden_dim
            # print(vocab_size, "vocab_size")
            # print(embedding_dim, "embedding_dim")
            # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

            # The linear layer that maps from hidden state space to tag space
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        def forward(self, sentence):


            sentence_reshaped = sentence.view((sentence.shape[0],1,-1))
            lstm_out, _ = self.lstm(sentence_reshaped)

            tag_space = self.hidden2tag(lstm_out.view(sentence.shape[0], -1))
            tag_scores = F.log_softmax(tag_space, dim=1)

            return tag_scores

    ######################################################
    ######################################################
    ######################################################

    # model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(ix_to_word), len(tag_to_ix))
    writer = SummaryWriter()

    print()
    print()
    print("Initializing Model")
    vocab_size = len(ix_to_word)
    tagset_size = len(tag_to_ix)
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, tagset_size)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    # with torch.no_grad():
    #     inputs = prepare_sequence_steps(training_data[0][0], ix_to_word)
    #     tag_scores = model(inputs)
    #     print(tag_scores)
    print()
    print()
    for epoch in range(EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
        for step_images, next_steps in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            step_images_in = prepare_sequence_steps(step_images)

            targets = prepare_sequence_steps(next_steps,long=True)

            # Step 3. Run our forward pass.
            next_steps_pred_scores = model(step_images_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            # print()

            # print(next_steps_pred_scores.shape,"Tag Scores Shape, Pre Loss")
            # print(targets.shape,"targets Shape, Pre Loss")
            loss = loss_function(next_steps_pred_scores, targets)
            loss.backward()
            writer.add_scalar('Loss/train', loss.detach().numpy(), epoch)
            optimizer.step()
            np_pred = np.argmax(next_steps_pred_scores.detach().numpy(),axis=1)
            np_targ = targets.detach().numpy()
            num_correct = 0
            for ix in range (np_pred.shape[0]):
                if np_pred[ix] == np_targ[ix]:
                    num_correct = num_correct + 1
            writer.add_scalar('Accuracy/train_acc', float(num_correct)/float(np_pred.shape[0]), epoch)
            writer.add_scalar('Accuracy/train_num_correct', num_correct, epoch)

            if epoch%50 ==0:
                print()
                print("/////////////////////////////////////////////////////////////////////////")
                print("/////////////////////////////////////////////////////////////////////////")
                print("Epoch",epoch)
                print()
                print("Predictions")
                for i in range(np_pred.shape[0]):
                    print(str(np_pred[i]).zfill(3),end=" ")
                print()
                print("Targets")
                for i in range(np_targ.shape[0]):
                    print(str(np_targ[i]).zfill(3),end=" ")
                print()
                print(loss)

            # if epoch%500 == 0:
            if epoch==200:
                # make_steps_images(step_images_in.detach().numpy(),"images/PredStep_"+str(epoch)+"_",PADDING*2+1,pred=np_pred)
                make_steps_images(step_images_in.detach().numpy(),"images/PredStep_Progress_",PADDING*2+1,pred=np_pred,targ=np_targ)

    print()
    print("End of Training")
    print()
    # See what the scores are after training
    with torch.no_grad():
        inputs = prepare_sequence_steps(training_data[0][0])
        tag_scores = model(inputs)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(tag_scores.shape)
        print(np.argmax(tag_scores,axis=1))
        print(targets)
        np_pred = np.argmax(next_steps_pred_scores.detach().numpy(),axis=1)
        np_targ = targets.detach().numpy()
        make_steps_images(inputs.detach().numpy(),"images/PredStep_Final_",PADDING*2+1,pred=np_pred,targ=np_targ)
    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
