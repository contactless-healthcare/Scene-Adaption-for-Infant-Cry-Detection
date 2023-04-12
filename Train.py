import config
import pickle
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import torch
from Model.model import FNN_Model, CNN_Model, SL_train_one_epoch, SCENE_Apdation_train_one_epoch, evaluate, DatasetLoad, AverageMeter, modelMetrics
from torch.utils.data import DataLoader



def Data_Acq(fileName):
    file = open(fileName, 'rb')
    sample = pickle.load(file, encoding='latin1')
    file.close()

    return sample




if __name__ == '__main__':
    CSD_Train_sample = Data_Acq(f".\\Data\\CSD_data.dat")
    train_data, train_label, evl_data, evl_label = CSD_Train_sample["trainSpectrogram"], CSD_Train_sample["trainLabel"], \
                                                   CSD_Train_sample["testSpectrogram"], CSD_Train_sample["testLabel"]

    if config.Data_Augmentation_Bool:
        aug_sample = Data_Acq(f".\\Data\\Crying_Github_Data_Aug.dat")
        dataAug_data_Github_Audioset, dataAug_label_Github_Audioset = aug_sample["spectrogram"], aug_sample["label"]
        dataAug_data_Audioset, dataAug_label_Audioset = aug_sample["audioset_spectrogram"], aug_sample["audioset_label"]

        train_data = np.concatenate([train_data, dataAug_data_Github_Audioset, dataAug_data_Audioset])
        train_label = np.concatenate([train_label, dataAug_label_Github_Audioset, dataAug_label_Audioset])

        # Class Balance
        rus = RandomUnderSampler(random_state=42)
        train_data_index = np.array([i for i in range(train_data.shape[0])])
        train_data_index = np.reshape(train_data_index, (-1, 1))
        train_data_index, train_label = rus.fit_resample(train_data_index, train_label)
        train_data_index = list(np.reshape(train_data_index, (1, -1)))
        train_data = train_data[train_data_index]

    # Acoustics
    # zNormalization = StandardScaler()
    # zNormalization.fit(train_data)
    # train_data = zNormalization.transform(train_data)
    # evl_data = zNormalization.transform(evl_data)

    # Spectrogram
    mean_train_data = np.mean(train_data)
    std_train_data = np.std(train_data)
    print("mean_train_data:", mean_train_data, "std_train_data:", std_train_data)

    train_data = (train_data - mean_train_data) / std_train_data
    evl_data = (evl_data - mean_train_data) / std_train_data


    ###########################################################################
    ########                   First Stage                               ######
    ###########################################################################
    # # SVM and KNN
    # parameters = {
    #     # SVM
    #     # 'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    #     # 'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
    #     # 'kernel': ['rbf']
    #
    #     # KNN
    #     "n_neighbors": [10, 20, 30, 40, 50]
    # }
    # # clf = GridSearchCV(SVC(probability=True), parameters, cv=5, n_jobs=8)
    # clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, n_jobs=8)
    # clf.fit(train_data, train_label)
    # print(clf.best_params_)
    # model = clf.best_estimator_
    #
    # pred_y = model.predict(evl_data)
    # evl_acc, evl_recall, evl_precision, evl_f1, evl_FPR, evl_confu_ma = modelMetrics(evl_label, pred_y)
    # # Save
    # with open(f".\\Model\\Weight\\KNN.pkl", 'wb') as f:
    #     pickle.dump(model, f)


    # FNN and CNN
    train_dataloader = DataLoader(
        dataset=DatasetLoad(train_data, train_label),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    evl_dataloader = DataLoader(
        dataset=DatasetLoad(evl_data, evl_label),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = "cuda"
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # model = FNN_Model().to(device)
    model = CNN_Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(config.SL_EPOCH):
        train_loss, train_acc, train_Recall, train_Precision, train_F1, train_FPR, train_confmat = \
            SL_train_one_epoch(model, optimizer, train_dataloader, device)
        evl_loss, evl_acc, evl_Recall, evl_Precision, evl_F1, evl_FPR, test_confmat = \
            evaluate(model, evl_dataloader, device)

        print(f"Epoch: {epoch + 1}/{config.SL_EPOCH}, train_loss: {train_loss:.6f}, train_acc: {train_acc:.6f}, "
              f"train_Recall: {train_Recall:.6f}, train_Precision: {train_Precision:.6f}, train_F1: {train_F1:.6f}, train_FPR: {train_FPR:.6f}")

        print(f"Epoch: {epoch + 1}/{config.SL_EPOCH}, test_loss : {evl_loss:.6f}, test_acc : {evl_acc:.6f}, "
              f"test_Recall : {evl_Recall:.6f}, test_Precision : {evl_Precision:.6f}, test_F1 : {evl_F1:.6f}, test_FPR : {evl_FPR:.6f} \n")

    _, evl_acc, evl_Recall, evl_Precision, evl_f1, evl_FPR, evl_confu_ma = evaluate(model, evl_dataloader, device)

    torch.save(model.state_dict(), f".\\Model\\Weight\\CNN_trained_on_CSD_without_scene_adaption.pkl")


    ###########################################################################
    ########                   Second Stage                             #######
    ###########################################################################
    accList, recallList, precisionList, f1List, FPRList, matrix_List  = [], [], [], [], [], []
    for dataDir in range(5):
        Test_sample = Data_Acq(f".\\Data\\UCD_{dataDir}_Fold.dat")
        test_data, test_label = Test_sample["testSpectrogram"], Test_sample["testLabel"]

        unlabeled_train_data, unlabled_train_label = Test_sample["trainAcoustic"], Test_sample["trainLabel"]

        # acoustics feature
        # test_data = zNormalization.transform(test_data)
        # unlabeled_train_data = zNormalization.transform(unlabeled_train_data)

        # Spectrogram
        test_data = (test_data - mean_train_data) / std_train_data
        unlabeled_train_data = (unlabeled_train_data - mean_train_data) / std_train_data

        # # SVM and KNN
        # with open(f".\\Model\\Weight\\KNN.pkl", 'rb') as f:
        #     model = pickle.load(f)
        #
        # IndexList = []
        # output = model.predict_proba(unlabeled_train_data)
        # probs = torch.nn.functional.softmax(torch.from_numpy(output), dim=1)
        # conf, classes = torch.max(probs, 1)
        # for i in range(conf.shape[0]):
        #     if conf[i] > config.confidence_threshold:
        #         IndexList.append(i)
        # unlabeled_train_data = unlabeled_train_data[IndexList]
        # unlabled_train_label = classes[IndexList]
        #
        # train_data = np.concatenate((train_data, unlabeled_train_data))
        # train_label = np.concatenate((train_label, unlabled_train_label))
        #
        # model.fit(train_data, train_label)

        # # FNN and CNN
        train_unlabeled_dataloader = DataLoader(
            dataset=DatasetLoad(unlabeled_train_data, unlabled_train_label),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        test_dataloader = DataLoader(
            dataset=DatasetLoad(test_data, test_label),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # model = FNN_Model().to(device)
        model = CNN_Model().to(device)
        model.load_state_dict(torch.load(f".\\Model\\Weight\\CNN_trained_on_CSD_without_scene_adaption.pkl"))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # EMA_model = FNN_Model().to(device)
        EMA_model = CNN_Model().to(device)
        EMA_model.load_state_dict(torch.load(f".\\Model\\Weight\\CNN_trained_on_CSD_without_scene_adaption.pkl"))

        # Scene adaption
        for epoch in range(config.SA_EPOCH):
            sl_loss, ssl_loss, all_loss, train_acc, train_Recall, train_Precision, train_F1, train_FPR, confmat = \
                SCENE_Apdation_train_one_epoch(EMA_model, model, optimizer, train_dataloader,
                                               train_unlabeled_dataloader, config.confidence_threshold, device)

            test_loss, test_acc, test_Recall, test_Precision, test_F1, test_FPR, test_confmat = \
                evaluate(model, test_dataloader, device)

            print(f"SA Epoch: {epoch + 1}/{config.SA_EPOCH}, Supervised loss: {sl_loss:.6f}, Unsupervised loss: {ssl_loss:.6f}")
            print(f"SA Epoch: {epoch + 1}/{config.SA_EPOCH}, train_loss: {all_loss:.6f}, train_acc: {train_acc:.6f}, "
                  f"train_Recall: {train_Recall:.6f}, train_Precision: {train_Precision:.6f}, train_F1: {train_F1:.6f}, train_FPR: {train_FPR:.6f}")
            print(f"SA Epoch: {epoch + 1}/{config.SA_EPOCH}, test_loss : {test_loss:.6f}, test_acc : {test_acc:.6f}, "
                  f"test_Recall : {test_Recall:.6f}, test_Precision : {test_Precision:.6f}, test_F1 : {test_F1:.6f}, test_FPR : {test_FPR:.6f} \n")

        torch.save(EMA_model.state_dict(), f".\\Model\\Weight\\EMA_Model_{dataDir}_with_SAF.pkl")
        torch.save(model.state_dict(), f".\\Model\\Weight\\Model_{dataDir}_with_SAF.pkl")

        ###########################################################################
        ########                   Evaluation Stage                             ###
        ###########################################################################

        # SVM, KNN
        # pred_y = model.predict(test_data)
        # acc, recall, precision, f1, FPR, confu_ma = modelMetrics(test_label, pred_y)

        # FNN and CNN
        _, acc, recall, precision, f1, FPR, confu_ma = evaluate(model, test_dataloader, device)

        accList.append(acc)
        recallList.append(recall)
        precisionList.append(precision)
        f1List.append(f1)
        FPRList.append(FPR)
        matrix_List.append(confu_ma)

    print(f"acc: {accList}")
    print(f"rec: {recallList}")
    print(f"pre: {precisionList}")
    print(f"f1 : {f1List}")
    print(f"FPR: {FPRList}")
    print("Acc mean:", np.mean(accList))
    print("Recall m:", np.mean(recallList))
    print("Precis m:", np.mean(precisionList))
    print("F1 mean :", np.mean(f1List))
    print("FPR mean:", np.mean(FPRList))
    for mat in matrix_List:
        print(mat, "\n")

    print("mean_train_data", mean_train_data)
    print("std_train_data", std_train_data)

    print("\n")
    print("ACC, RECALL, PRECSIION, F1, FPR")
    print(f"{np.mean(accList) * 100:.4f}")
    print(f"{np.mean(recallList) * 100:.4f}")
    print(f"{np.mean(precisionList) * 100:.4f}")
    print(f"{np.mean(f1List) * 100:.4f}")
    print(f"{np.mean(FPRList) * 100:.4f}")



