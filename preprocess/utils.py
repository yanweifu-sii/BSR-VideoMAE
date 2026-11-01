import pickle
import numpy as np
import os

def save_eeg(X, label, channel_order, save_path):
    eeg = {
        channel_order[i].upper(): X[i, :] for i in range(len(channel_order))
    }
    pickle.dump(
        {"eeg": eeg, "label": label},
        open(save_path, "wb"),
    )
    
def create_folder_eeg(root, subjects=0, splits=["train", "val", "test"]):
    if subjects > 0: 
        # set each subject as independent subset
        # dataset - subject ID - train/val/test
        for subject in range(subjects):
            for split in splits:
                split_path = os.path.join(root, "subject{}".format(str(subject+1)), split)
                if not os.path.isdir(split_path):
                    os.makedirs(split_path)
    else:
        # dataset - train/val/test
        for split in splits:
            split_path = os.path.join(root, split)
            if not os.path.isdir(split_path):
                os.makedirs(split_path)

def read_eeg(save_path):
    sample = pickle.load(open(save_path, "rb"))
    eeg = sample["eeg"]
    label = sample["label"]
    return eeg, label
        
# test
if __name__ == "__main__":
    # X = np.random.rand(2, 100)
    # channel_order = ["fp1", "fp2"]
    # label = "1"
    # save_path = "/path/to/data/test.pkl"
    # save_eeg(X, label, channel_order, save_path)
    
    print(read_eeg("/path/to/data/eeg/tueg/train/sample.pkl"))
    
    # create_folder_eeg(root="/path/to/data/eeg/shumi")
    