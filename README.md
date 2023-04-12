# A-Scene-Adaption-Framework-for-Infant-Cry-Detection-in-Obstetrics

This is code repository for the paper "A Scene Adaption Framework for Infant Cry Detection in Obstetrics".


It is developed for Infant Cry detection in a new environment (including clinical setting, e.g. Obstetrics).


There are six major .py files in this repo, which sequential runs implement infant cry detection based on machine learning:

    1. Config.py: It controls the required parameters in each process.
    2. Preprocessing.py: It contains the data segmentation to split the raw record to the segmentation with the predetermined length.
    3. Feature extraction.py: It transforms the lung sound signal into an input suitable for the model, such as statistical features and spectrogram.
    4. Scene Simulation Data Augmentation.py: It realizes the scene simulation-based data augmentation for infant cry detection.
    5. Train.py: It consists of two training phases, a supervised training phase based on labeled data from source domain and a unsupervised training phase (scene adaptation) based on unlabeled data from target domain.
    6. Test Demo for record: It provides the demonstration of making predictions to raw records.
    
    
In addition, we provide the trained CNN model for more extensive validation.



Please cite the below paper if the code was used in your research or development.
    
    @inproceedings{
        title={A Scene Adaption Framework for Infant Cry Detection in Obstetrics},
        author={Huang, Dongmin and Ren, Lirong and Lu, Hongzhou and Wang, Wenjin}, 
        booktitle={2023 45th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
        year={2023}
    }


