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
from ModelFunctions import *

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
PADDING = 20
EMBEDDING_DIM = (PADDING*2+1)*(PADDING*2+1) # N_Features
NUM_CLASSES = (PADDING*2+1)*(PADDING*2+1)+1 # Bonus Class is for the end of track class
EPOCHS = 5000
# INFILE = "/home/jmills/workdir/TrackWalker/inputfiles/merged_dlreco_75e9707a-a05b-4cb7-a246-bedc2982ff7e.root"
INFILE = "/home/jmills/workdir/TrackWalker/inputfiles/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtmerge_TrackWalker_traindata_198files.root"
REAL_IM = True
TRACK_IDX = 0
EVENT_IDX = 0
ALWAYS_EDGE = True # True points are always placed at the edge of the Padded Box
DO_TENSORLOG = True
TENSORDIR = None # Default runs/DATE_TIME
# TENSORDIR = "runs/Pad20_Hidden1024_500Entries"
CLASSIFIER_NOT_DISTANCESHIFTER = True # True -> Predict Output Pixel to step to, False -> Predict X,Y shift to next point
NDIMENSIONS = 2 #Not configured to have 3 yet.
LEARNING_RATE = 0.01 # 0.01 is good for the classifier mode,
TRAINING_SIZE = -1 # -1 Means do all, otherwise only include first N training examples
NENTRIES_TO_DO = 1000 # Entries to read from ROOT File


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
    training_data = []
    event_ids = []
    step_dist_3d = []
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
        image_list,xs,ys, runs, subruns, events, filepaths, entries, track_pdgs = load_rootfile(INFILE, step_dist_3d, nentries_to_do = NENTRIES_TO_DO)
        print()
        print()
        # save_im(image_list[EVENT_IDX],"images/EventDisp")
        for EVENT_IDX in range(len(image_list)):
            print("Doing Event:", EVENT_IDX)
            print("N MC Tracks:", len(xs[EVENT_IDX]))
            for TRACK_IDX in range(len(xs[EVENT_IDX])):
                # if TRACK_IDX != 0:
                #     continue
                print("     Doing Track:", TRACK_IDX)

                full_image = image_list[EVENT_IDX]
                steps_x = xs[EVENT_IDX][TRACK_IDX]
                steps_y = ys[EVENT_IDX][TRACK_IDX]

                print("         Original Track Points", len(steps_x))
                new_steps_x, new_steps_y = insert_cropedge_steps(steps_x,steps_y,PADDING,always_edge=ALWAYS_EDGE)
                steps_x = new_steps_x
                steps_y = new_steps_y

                print("         After Inserted Track Points", len(steps_x))
                if len(steps_x) == 0: #Don't  include tracks without points.
                    continue
                # save_im(full_image,'file',canv_x = 4000, canv_y = 1000) # If you want to save the full image as an event_display

                # Many of the following categories are just a reformatting of each other
                # They are duplicated to allow for easy network mode switching
                stepped_images = [] # List of cropped images as 2D numpy array
                flat_stepped_images = [] # list of cropped images as flattened 1D np array
                next_positions = [] # list of next step positions as np(x,y)
                flat_next_positions = [] # list of next step positions in flattened single coord idx
                xy_shifts = [] # list of X,Y shifts to take the next step
                for idx in range(len(steps_x)):
                    # if idx > 1:
                    #     continue
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
                        np_xy_shift = np.array([target_x*1.0-PADDING,target_y*1.0-PADDING ])
                        xy_shifts.append(np_xy_shift)
                    else:
                        next_positions.append(np.array([-1.0,-1.0]))
                        flat_next_positions.append(NUM_CLASSES-1)
                        np_xy_shift = np.array([0.0,0.0])
                        xy_shifts.append(np_xy_shift)
                if CLASSIFIER_NOT_DISTANCESHIFTER:
                    training_data.append((flat_stepped_images,flat_next_positions))
                    event_ids.append(EVENT_IDX)
                else:
                    training_data.append((flat_stepped_images,xy_shifts))
                    event_ids.append(EVENT_IDX)
    cava = ROOT.TCanvas("c","c",1000,800)
    h_3d = ROOT.TH1D("h3d","h3d",50,0,50)
    min_t = 0
    for d in step_dist_3d:
        h_3d.Fill(d)
    h_3d.Draw()
    cava.SaveAs("images/Step3dDist.png")

    if TRAINING_SIZE != -1:
        training_data[0:TRAINING_SIZE]
    print("Number of Training Examples:", len(training_data))

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




    # ix_to_word = {}
    # # For each words-list (sentence) and tags-list in each tuple of training_data
    # for sent, tags in training_data:
    #     for ix,flat_stepped_image in enumerate(sent):
    #         is_in = False
    #         for iii in range(len(ix_to_word)):
    #             if np.array_equal(flat_stepped_image, ix_to_word[iii]):
    #                 print("--------------------------------------------")
    #                 print("Is already in.")
    #                 print("--------------------------------------------")
    #                 is_in = True
    #                 break
    #         if is_in == False:
    #             ix_to_word[len(ix_to_word)] = flat_stepped_image  # Assign each word with a unique index



    ######################################################
    ######################################################
    ######################################################

    class LSTMTagger(nn.Module):

        def __init__(self, embedding_dim, hidden_dim, output_dim):
            super(LSTMTagger, self).__init__()
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, output_dim)

        def forward(self, sentence):
            sentence_reshaped = sentence.view((sentence.shape[0],1,-1))
            lstm_out, _ = self.lstm(sentence_reshaped)
            tag_space = self.hidden2tag(lstm_out.view(sentence.shape[0], -1))
            tag_scores = F.log_softmax(tag_space, dim=1)

            return tag_scores

    ######################################################
    ######################################################
    ######################################################

    writer = []
    if DO_TENSORLOG:
        if TENSORDIR == None:
            writer = SummaryWriter()
        else:
            writer = SummaryWriter(log_dir=TENSORDIR)
    print()
    print()
    print("Initializing Model")
    output_dim = None
    loss_function = None
    if CLASSIFIER_NOT_DISTANCESHIFTER:
        output_dim = input_image_dimension*input_image_dimension+1 # nPixels in crop + 1 for 'end of track'
        loss_function = nn.NLLLoss(reduction='none')
    else:
        output_dim = NDIMENSIONS # Shift X, Shift Y
        loss_function = nn.MSELoss(reduction='sum')

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, output_dim)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print()
    print()
    step_counter = 0
    for epoch in range(EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
        if epoch%50 ==0:
            print()
            print("/////////////////////////////////////////////////////////////////////////")
            print("/////////////////////////////////////////////////////////////////////////")
            print("Epoch",epoch)
        train_idx = -1



        epoch_loss_average = 0
        epoch_train_acc_exact = 0
        epoch_train_acc_2dist = 0
        epoch_train_acc_5dist = 0
        epoch_train_num_correct_exact = 0
        epoch_train_average_distance_off = 0

        for step_images, next_steps in training_data:
            step_counter += 1
            train_idx += 1
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            step_images_in = prepare_sequence_steps(step_images)
            is_long = CLASSIFIER_NOT_DISTANCESHIFTER
            targets = prepare_sequence_steps(next_steps,long=is_long)
            np_targ = targets.detach().numpy()

            # Step 3. Run our forward pass.
            next_steps_pred_scores = model(step_images_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            # print()

            # print(next_steps_pred_scores.shape,"Tag Scores Shape, Pre Loss")
            # print(targets.shape,"targets Shape, Pre Loss")

            np_pred = None
            if CLASSIFIER_NOT_DISTANCESHIFTER:
                np_pred = np.argmax(next_steps_pred_scores.detach().numpy(),axis=1)
            else:
                np_pred = np.rint(next_steps_pred_scores.detach().numpy()) # Rounded to integers



            loss_weights = torch.tensor(get_loss_weights_v2(targets.detach().numpy(),np_pred,PADDING*2+1),dtype=torch.float)
            loss = loss_function(next_steps_pred_scores, targets)
            loss_weighted = loss*loss_weights
            loss_total = torch.mean(loss_weighted)
            loss_total.backward()
            optimizer.step()

            num_correct_exact = 0
            num_correct_2dist = 0
            num_correct_5dist = 0
            dists = np.zeros(np_pred.shape[0])

            for ix in range(np_pred.shape[0]):
                targ_x, targ_y = unflatten_pos(np_targ[ix], input_image_dimension)
                pred_x, pred_y = unflatten_pos(np_pred[ix], input_image_dimension)
                dists[ix] = ((pred_x - targ_x)**2 + (pred_y - targ_y)**2)**0.5
                if CLASSIFIER_NOT_DISTANCESHIFTER:
                    if np_pred[ix] == np_targ[ix]:
                        num_correct_exact = num_correct_exact + 1
                    if dists[ix] <= 2.0:
                        num_correct_2dist += 1
                    if dists[ix] <= 5.0:
                        num_correct_5dist += 1
                else:
                    if np.array_equal(np_pred[ix], np_targ[ix]):
                        num_correct_exact += 1
            if DO_TENSORLOG:
                writer.add_scalar('Steps/train_loss', loss_total.detach().numpy(), step_counter)
                writer.add_scalar('Steps/train_acc_exact', float(num_correct_exact)/float(np_pred.shape[0]), step_counter)
                writer.add_scalar('Steps/train_acc_2dist', float(num_correct_2dist)/float(np_pred.shape[0]), step_counter)
                writer.add_scalar('Steps/train_acc_5dist', float(num_correct_5dist)/float(np_pred.shape[0]), step_counter)
                writer.add_scalar('Steps/train_num_correct_exact', num_correct_exact, step_counter)
                writer.add_scalar("Steps/train_average_off_distance", np.mean(dists),step_counter)
                epoch_loss_average += loss_total.detach().numpy()/len(training_data)
                epoch_train_acc_exact += float(num_correct_exact)/float(np_pred.shape[0])/len(training_data)
                epoch_train_acc_2dist += float(num_correct_2dist)/float(np_pred.shape[0])/len(training_data)
                epoch_train_acc_5dist += float(num_correct_5dist)/float(np_pred.shape[0])/len(training_data)
                epoch_train_num_correct_exact += num_correct_exact/len(training_data)
                epoch_train_average_distance_off += np.mean(dists)/len(training_data)


            # if epoch%500 == 0:
            # if epoch==200:
                # make_steps_images(step_images_in.detach().numpy(),"images/PredStep_"+str(epoch)+"_",PADDING*2+1,pred=np_pred)
            # make_steps_images(step_images_in.detach().numpy(),"images/PredStep_Progress_",PADDING*2+1,pred=np_pred,targ=np_targ)
        if epoch%50 ==0:
            print("Epoch Averaged")
            print("Exact Accuracy:")
            print(epoch_train_acc_exact)
            print("Within 2 Accuracy:")
            print(epoch_train_acc_2dist)
            print("Within 5 Accuracy:")
            print(epoch_train_acc_5dist)
            print("Loss:")
            print(epoch_loss_average)
            print("/////////////////////////////")
            print()
        if DO_TENSORLOG:
            writer.add_scalar('Epoch/train_loss', epoch_loss_average, epoch)
            writer.add_scalar('Epoch/train_acc_exact', epoch_train_acc_exact, epoch)
            writer.add_scalar('Epoch/train_acc_2dist', epoch_train_acc_2dist, epoch)
            writer.add_scalar('Epoch/train_acc_5dist', epoch_train_acc_5dist, epoch)
            writer.add_scalar('Epoch/train_num_correct_exact', epoch_train_num_correct_exact, epoch)
            writer.add_scalar("Epoch/train_average_off_distance", epoch_train_average_distance_off,epoch)
        if epoch%2000 == 0:
            torch.save(model.state_dict(), "model_checkpoints/TrackerCheckPoint_"+str(epoch)+".pt")
    print()
    print("End of Training")
    print()
    # See what the scores are after training
    with torch.no_grad():
        train_idx = -1
        for step_images, next_steps in training_data:
            train_idx += 1
            print()
            print("Event:", event_ids[train_idx])
            print("Track Idx:",train_idx)
            step_images_in = prepare_sequence_steps(step_images)

            targets = prepare_sequence_steps(next_steps,long=is_long)
            np_targ = targets.detach().numpy()

            next_steps_pred_scores = model(step_images_in)

            np_pred = None
            if CLASSIFIER_NOT_DISTANCESHIFTER:
                np_pred = np.argmax(next_steps_pred_scores.detach().numpy(),axis=1)
            else:
                np_pred = np.rint(next_steps_pred_scores.detach().numpy()) # Rounded to integers
            torch.save(model.state_dict(), "model_checkpoints/TrackerCheckPoint_"+str(EPOCHS)+"_Fin.pt")



            num_correct_exact = 0
            for ix in range(np_pred.shape[0]):
                if CLASSIFIER_NOT_DISTANCESHIFTER:
                    if np_pred[ix] == np_targ[ix]:
                        num_correct_exact = num_correct_exact + 1
                else:
                    if np.array_equal(np_pred[ix], np_targ[ix]):
                        num_correct_exact += 1
            print("Accuracy",float(num_correct_exact)/float(np_pred.shape[0]))
            print("Points:",float(np_pred.shape[0]))

            np_targ = targets.detach().numpy()
            if not CLASSIFIER_NOT_DISTANCESHIFTER:
                print("Predictions Raw")
                print(next_steps_pred_scores.detach().numpy())

            # make_steps_images(step_images_in.detach().numpy(),"images/PredStep_Final_"+str(train_idx).zfill(2)+"_",PADDING*2+1,pred=np_pred,targ=np_targ)
    print("End of Main")
    return 0

if __name__ == '__main__':
    main()
