# Scene-Adaption-for-Infant-Cry-Detection
This is code repository for the paper "A Scene Adaption Framework for Infant Cry Detection in Obstetrics", which had been accpted by 2023 EMBC.


There are six major .py files in this repo.

    Config.py: It controls the required parameters in each process.
    Preprocessing.py: It contains the data segmentation.
    Feature extraction.py: It transforms the sound signals into an input suitable for the model, such as statistical features and spectrogram.
    Scene_Simulation_Data_augmentation.py: It has the implementation of scene simulation-based data augmentation.
    Train.py: It is used to train and test the model.
    Test_Demo_for_record.py: It provides examples from the original recording to predicting crying.

We provide the weights of CNN for more extensive testing.

Please cite below paper if the code was used in your research or development.

    @inproceedings{
        title={A Scene Adaption Framework for Infant Cry Detection in Obstetrics},
        author={Huang, Dongmin and Ren, Lirong and Lu, Hongzhou and Wang, Wenjin},
        booktitle={2023 45th Annual International Conference of the IEEE Engineering in Medicine and Biology Society},
        year={2023}
    }

