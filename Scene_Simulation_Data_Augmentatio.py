import config
import pickle
import numpy as np
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import Feature_Extraction


def Data_Acq(fileName):
    file = open(fileName, 'rb')
    sample = pickle.load(file, encoding='latin1')
    file.close()

    return sample



def Scene_Simulation_Data_Augmentatio(github_train_data, github_train_label, audioset_data, audioset_label):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])

    audioset_data_IndexList = [i for i in range(audioset_data.shape[0])]
    for i in audioset_data_IndexList:
        audioset_data[i] = Feature_Extraction.dc_normalize(audioset_data[i])

    aug_data_label = {
        "data": [],
        "label": [],
        "acoustic_feature": [],
        "spectrogram": [],

        "audioset_data": [],
        "audioset_label": [],
        "audioset_acoustic_feature": [],
        "audioset_spectrogram": []
    }

    readyAdd_IndexList = []
    for i in range(github_train_data.shape[0]):
        github_train_data[i] = augment(samples=github_train_data[i], sample_rate=config.audioSampleRate)
        github_train_data[i] = Feature_Extraction.dc_normalize(github_train_data[i])

        # The probability of adding directly
        adding_directly_probability = random.uniform(0, 1)
        adding_Number = random.randint(2, 3)
        audioset_Chosen_Index = [np.random.choice(audioset_data_IndexList, size=1)[0] for _ in range(adding_Number)]
        readyAdd_IndexList = readyAdd_IndexList + audioset_Chosen_Index

        if adding_directly_probability > 0.5:
            aug_data = github_train_data[i]
            for Chosen_Index in audioset_Chosen_Index:
                aug_data += audioset_data[Chosen_Index]

            aug_data = Feature_Extraction.dc_normalize(aug_data)
        else:
            different_Length = [random.uniform(1, config.slidingWindows)*config.audioSampleRate   for _ in range(adding_Number)]

            aug_data = github_train_data[i][:int(different_Length[0])]
            padding = np.zeros((int(config.audioSampleRate * config.slidingWindows -len(aug_data)), ))
            aug_data = np.concatenate((aug_data, padding))

            Chosen_Index = 0
            for dif_Length in different_Length[1:]:
                temp_data = audioset_data[audioset_Chosen_Index[Chosen_Index]][:int(dif_Length)]

                strat_Time = int(random.uniform(0, config.slidingWindows)*config.audioSampleRate)
                if strat_Time + len(temp_data) > config.slidingWindows * config.audioSampleRate:
                    end_Time = int(config.slidingWindows * config.audioSampleRate - strat_Time)
                    aug_data[strat_Time:] += temp_data[:end_Time]
                else:
                    end_Time = strat_Time + len(temp_data)
                    aug_data[strat_Time:end_Time] += temp_data

            aug_data = Feature_Extraction.dc_normalize(aug_data)    # goal

        # write(f"goal.wav", config.audioSampleRate, goal)
        # fig, axs = plt.subplots(3, 2)
        # axs[0, 0].plot(github_train_data[i])
        # axs[1, 0].plot(aug_data)
        # axs[2, 0].plot(goal)
        # plt.show()
        # print()

        goal_label = 0
        if github_train_label[i] != 0:  # Cry - 0, No-Cry - 1
            goal_label += 1

        acoustic_feature, spectrogram_feature = Feature_Extraction.acoustic_features_and_spectrogram(np.reshape(aug_data, (1, -1)))

        aug_data_label["data"].append(aug_data)
        aug_data_label["label"].append(goal_label)
        aug_data_label["acoustic_feature"].append(acoustic_feature)
        aug_data_label["spectrogram"].append(spectrogram_feature)

    aug_data_label["data"] = np.array(aug_data_label["data"])
    aug_data_label["label"] = np.array(aug_data_label["label"])
    aug_data_label["acoustic_feature"] = np.array(aug_data_label["acoustic_feature"]).squeeze(1)
    aug_data_label["spectrogram"] = np.array(aug_data_label["spectrogram"]).squeeze(1)


    for i in audioset_data_IndexList:
        if i not in readyAdd_IndexList:
            acoustic_feature, spectrogram_feature = Feature_Extraction.acoustic_features_and_spectrogram(np.reshape(audioset_data[i], (1, -1)))

            aug_data_label["audioset_data"].append(audioset_data[i])
            aug_data_label["audioset_label"].append(audioset_label[i])
            aug_data_label["audioset_acoustic_feature"].append(acoustic_feature)
            aug_data_label["audioset_spectrogram"].append(spectrogram_feature)

    aug_data_label["audioset_data"] = np.array(aug_data_label["audioset_data"])
    aug_data_label["audioset_label"] = np.array(aug_data_label["audioset_label"])
    aug_data_label["audioset_acoustic_feature"] = np.array(aug_data_label["audioset_acoustic_feature"]).squeeze(1)
    aug_data_label["audioset_spectrogram"] = np.array(aug_data_label["audioset_spectrogram"]).squeeze(1)

    return aug_data_label




def main():
    sampleDir = f"Data\\Crying_Github.dat"
    sample = Data_Acq(sampleDir)
    github_train_data, github_train_label = sample["data"], sample["label"]
    audioset_data, audioset_label = sample["data"], sample["label"]
    audioset_data = np.concatenate((audioset_data, audioset_data, audioset_data, audioset_data, audioset_data, audioset_data))
    audioset_label = np.concatenate((audioset_label, audioset_label, audioset_label, audioset_label, audioset_label, audioset_label))

    aug_sample = Scene_Simulation_Data_Augmentatio(github_train_data, github_train_label, audioset_data, audioset_label)

    # Save
    saveDir = "Data\\Aug_Sample.dat"
    file = open(saveDir, 'wb')
    pickle.dump(aug_sample, file)
    file.close()

    # saveDir = "Data\\Aug_Sample.dat"
    # aug_sample = Data_Acq(saveDir)
    # print(aug_sample)


if __name__ == "__main__":
    main()






