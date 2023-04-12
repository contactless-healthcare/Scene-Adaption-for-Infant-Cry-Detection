import numpy as np
import config
import librosa
import librosa.display
import pickle


def Data_Acq(fileName):
    file = open(fileName, 'rb')
    sample = pickle.load(file, encoding='latin1')
    file.close()

    return sample



def Wav2Segments(dataDir, labelDir):
    def SlidingWindows2Segments(CryList, raw_audio, sample_rate, slidingLengthWindows, overlap, start=0):
        slidingLengthWindows = int(slidingLengthWindows * sample_rate)  # number of frame
        overlap = int(overlap * sample_rate)  # number of overlap

        sample = {
            "data": [],
            "label": []
        }
        while start < len(raw_audio):
            end = int(start + slidingLengthWindows)
            # Data Acq
            if end > len(raw_audio):
                data = raw_audio[start: -1]
                padding = np.zeros((end-len(raw_audio)+1,))
                data = np.concatenate((data, padding))
            else:
                data = raw_audio[start: end]

            # Label set
            label = 1       # Cry - 0, No-Cry - 1
            frameWork = [i for i in range(start, end)]
            if len(list(set(frameWork).intersection(set(CryList)))) != 0:
                label -= 1

            # 存储
            sample["data"].append(data)
            sample["label"].append(label)

            start += overlap

        sample["data"] = np.array(sample["data"])
        sample["label"] = np.array(sample["label"])

        return sample


    raw_audio, sample_rate = librosa.load(path=dataDir, sr=config.audioSampleRate)

    # Label from file
    CryList = None
    f = open(f"{labelDir}", "r")
    res = f.readlines()
    for cur in res:
        cur = cur.strip("\n")
        lStart, lEnd, lSet = cur.split("\t")

        lStart = int(float(lStart) * config.audioSampleRate)
        lEnd = int(float(lEnd) * config.audioSampleRate)

        if int(lSet) == 0:
            if CryList is None:
                CryList = np.array([i for i in range(lStart, lEnd)])
            else:
                CryList = np.concatenate((CryList, np.array([i for i in range(lStart, lEnd)])))
    if CryList is None:
        CryList = []

    # Sliding windows to traim audio
    sample = SlidingWindows2Segments(CryList, raw_audio, sample_rate, config.slidingWindows, config.overlap, start=0)

    # Save
    saveDir = dataDir.split(".")[0] + ".dat"
    file = open(saveDir, 'wb')
    pickle.dump(sample, file)
    file.close()

    return sample




def main():
    dataDir = f"Data\\Crying_Github.wav"
    labelDir = f"Data\\Crying_Github.txt"
    Wav2Segments(dataDir, labelDir)

    # sampleDir = f"Data\\Crying_Github.dat"
    # sample = Data_Acq(sampleDir)
    # print(sample)



if __name__ == '__main__':
    main()






