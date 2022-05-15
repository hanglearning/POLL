import numpy as np
from torchvision import datasets, transforms


def get_dataset_cifar10_extr_noniid(n_devices, n_class, nsamples, rate_unbalance):
    data_dir = './data'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)

    # Chose equal splits for every user
    user_groups_train, user_groups_test = cifar_extr_noniid(
        train_dataset, test_dataset, n_devices, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test


def cifar_extr_noniid(train_dataset, test_dataset, n_devices, n_class, num_samples, rate_unbalance):
    num_shards_train, num_imgs_train = int(50000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000

    assert(n_class * n_devices <= num_shards_train)
    assert(n_class <= num_classes)

    idx_class = [i for i in range(num_classes)]
    idx_shard = np.array([i for i in range(num_shards_train)])

    dict_users_train = {i: np.array([]) for i in range(n_devices)} # store indexes of data samples
    dict_users_test = {i: np.array([]) for i in range(n_devices)}
    
    idxs = np.arange(num_shards_train*num_imgs_train)
    labels = np.array(train_dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] # each run, the same
    idxs = idxs_labels[0, :]
    # so shuffle it, then idxs[rand*num_imgs_train:(rand+1)*num_imgs_train] becomes more random
    starting_index = 0
    for _ in range(num_classes):
        np.random.shuffle(idxs[starting_index:starting_index + int(len(idxs)/num_classes)])
        starting_index += int(len(idxs)/num_classes)
    labels = idxs_labels[1, :] # sorted by label 0 ~ 9

    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()] # each run, the same
    idxs_test = idxs_labels_test[0, :]
    # shuffle for the same reason
    starting_index = 0
    for _ in range(num_classes):
        np.random.shuffle(idxs_test[starting_index:starting_index + int(len(idxs_test)/num_classes)])
        starting_index += int(len(idxs_test)/num_classes)
    labels_test = idxs_labels_test[1, :]

    idxs_test_splits = [[] for i in range(num_classes)]
    for i in range(len(labels_test)):
        idxs_test_splits[labels_test[i]].append(idxs_test[i])
        # idxs_test_splits has 10 lists, each represents a class
        # each list now appends an index of the data point of that class in the test set
        # overall, idxs_test_splits is like a dictionary, key:value -> class: indexes_of_that_class_in_test_set

    # correspond to sorted labels 0 ~ 9
    idx_shards = np.split(idx_shard, 10)

    # divide and assign
    for i in range(n_devices):
        user_labels = np.array([])
        temp_set = set(np.random.choice(10, n_class, replace=False)) # get temp labels for this user
        rand_set = []
        for j in temp_set:
            choice = np.random.choice(idx_shards[j], 1)[0] # index of "the label index (corresponding to data sample as well)"
            rand_set.append(int(choice))
            idx_shards[j] = np.delete(
                idx_shards[j], np.where(idx_shards[j] == choice))
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0) 
                    # max rand can be 2499, and idxs is sorted. but argsort makes idxs the same for each execution. better shuffle idxs again
                user_labels = np.concatenate(
                    (user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate(
                    (user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)

        # test set has all data samples of that label
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test_splits[int(label)]), axis=0)
    return dict_users_train, dict_users_test
