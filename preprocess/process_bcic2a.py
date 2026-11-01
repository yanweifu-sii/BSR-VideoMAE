import os
import glob
import numpy as np
import scipy.io as sio
import sys
import mne
from utils import save_eeg, create_folder_eeg

'''
    This code is part of the fbcsptoolbox,which is a codebase in the github.
    Thanks for the https://fbcsptoolbox.github.io/ to provie this code.
'''

DATA_ROOT = os.environ.get("BSR_DATA_ROOT", "/eeg")
DST_ROOT = os.path.join(DATA_ROOT, "bcic2a_fewshot")

channel_order = ["FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ"]

class LoadData:
    def __init__(self, eeg_file_path: str):
        self.eeg_file_path = eeg_file_path
        self.raw_eeg_subject = None

    def load_raw_data_gdf(self, file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self,file_to_load):
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self, file_path_extension: str = None):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)
    def filter(self,low,high):
        self.raw_eeg_subject.filter(low,high)

class LoadBCIC(LoadData):
    """Subclass of LoadData for loading BCI Competition IV Dataset 2a"""
    def __init__(self, file_to_load, *args):
        self.stimcodes = ['769', '770', '771', '772']
        # self.epoched_data={}
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        self.fs = None
        super(LoadBCIC, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, bandpass = None,resample = None,baseline=None,reject = False):
        
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        events, event_ids = mne.events_from_annotations(raw_data)
        self.fs = raw_data.info.get('sfreq')
        if reject == True:
            reject_events = mne.pick_events(events,[1])
            reject_oneset = reject_events[:,0]/self.fs
            duration = [4]*len(reject_events)
            descriptions = ['bad trial']*len(reject_events)
            blink_annot = mne.Annotations(reject_oneset,duration,descriptions)
            raw_data.set_annotations(blink_annot)
        
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=True)
        if bandpass is not None:
            epochs.filter(bandpass[0],bandpass[1],method = 'iir')
            # epochs.resample(128)
        if resample is not None:
            epochs.resample(resample)

        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs
                  }
        return eeg_data
    
class LoadBCIC_E(LoadData):
    """A class to lode the test data of the BICI IV 2a dataset"""
    def __init__(self, file_to_load, lable_name, *args):
        self.stimcodes = ('783')
        # self.epoched_data={}
        self.label_name = lable_name # the path of the test label
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(LoadBCIC_E, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, bandpass = False,resample = None,baseline=None):
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)



        if bandpass is not None:
            epochs.filter(bandpass[0],bandpass[1],method = 'iir')
            # epochs.resample(128)
        if resample is not None:
            epochs.resample(resample)
        epochs = epochs.drop_channels(self.channels_to_remove)
        label_info  = sio.loadmat(os.path.join(self.eeg_file_path,self.label_name))
        #label_info shape:(288, 1)
        self.y_labels = label_info['classlabel'].reshape(-1) -1
        # print(self.y_labels)
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs}
        return eeg_data
    
# def EMS(data):
#     new_x = []

#     for x in data:
#         new_x.append(exponential_moving_standardize(x))

#     return np.array(new_x)


def Load_BCIC_2a_raw_data(tmin=0, tmax=4,bandpass = [0,38],resample = None):
    '''
    Load the BCIC 2a data.
    Arg:
        sub:The subject whose data need to be load.
    '''

    data_path = os.environ.get("BCIC2A_RAW", "/bcic-iv/2a")
    for sub in range(1,10):
        data_name = r'A0{}T.gdf'.format(sub)
        data_loader = LoadBCIC(data_name, data_path)
        data = data_loader.get_epochs(tmin=tmin, tmax=tmax,bandpass = bandpass,resample = resample)
        print('orgin size is:',data['x_data'].shape)
        train_x = np.array(data['x_data'])[:, :, :]
        train_y = np.array(data['y_labels'])
        
        data_name = r'A0{}E.gdf'.format(sub)
        label_name = r'A0{}E.mat'.format(sub)
        data_loader = LoadBCIC_E(data_name, label_name, data_path)
        data = data_loader.get_epochs(tmin=tmin, tmax=tmax,bandpass = bandpass,resample = resample)
        test_x = np.array(data['x_data'])[:, :, :]
        test_y = data['y_labels']
        

        train_x = np.array(train_x)
        train_y = np.array(train_y).reshape(-1)

        
        test_x = np.array(test_x)
        test_y = np.array(test_y).reshape(-1)
 
        print('trian_x:',train_x.shape)
        print('train_y:',train_y.shape)
        
        print('test_x:',test_x.shape)
        print('test_y:',test_y.shape)
        
        train_x = train_x[:, :, :800]
        n_train, _, _ = train_x.shape
        for i in range(n_train):
            X = train_x[i]
            y = train_y[i]
            save_eeg(X=X, label=y, channel_order=channel_order, 
                     save_path=os.path.join(DST_ROOT, "subject"+str(sub), "train", r'{}_{}.pkl'.format(sub, i))
                     )
        
        test_x = test_x[:, :, :800]
        n_test, _, _ = test_x.shape
        for i in range(n_test):
            X = test_x[i]
            y = test_y[i]
            save_eeg(X=X, label=y, channel_order=channel_order, 
                     save_path=os.path.join(DST_ROOT, "subject"+str(sub), "test", r'{}_{}.pkl'.format(sub, i))
                     )
            
create_folder_eeg(DST_ROOT, subjects=9, splits=["train", "test"])
Load_BCIC_2a_raw_data(resample=200)
            