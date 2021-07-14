import numpy as np
from skimage.io import imread
import os
from sklearn.neighbors import KNeighborsClassifier
import math
import matplotlib.pyplot as plt

data = np.zeros(shape=(1, 10304))
non_faces = np.zeros(shape=(1, 10304))


def scan_folder(parent, format):
    global data
    # iterate over all the files in directory 'parent'
    for file_name in sorted(os.listdir(parent), key=len):
        if file_name.endswith(format):
            # if it's a txt file, print its name (or do whatever you want)
            file_path = parent + '/' + file_name
            image = np.array(imread(file_path)).flatten()
            data = np.append(data, np.matrix(image), axis=0)

        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recursively call this method
                scan_folder(current_path, format)


def comp_eigens(d):
    mean_values = np.mean(d, axis=0)
    Z = d - mean_values
    cov = np.cov(Z, rowvar=False, bias=1)
    return np.linalg.eigh(cov)


def PCA(alpha, eig_values, eig_vectors):
    r = 0
    s = np.sum(eig_values)
    explained_variance = 0.0
    for x in range(len(eig_values)):
        if explained_variance < alpha:
            explained_variance += eig_values[x] / s
        else:
            r = x
            break
    P = eig_vectors[:, :r]
    return P


def LDA(d, test, labels):
    Ur = []
    tempZ = np.zeros((5, 10304))
    S = np.zeros((1, 10304))
    sb = np.zeros((1, 10304))
    mean_values = np.mean(d, axis=0)
    means = np.zeros((1, 10304))
    Z = np.zeros((1, 10304))
    for i in range(40):
        means = np.append(means, np.mean(d[5 * i:5 + 5 * i, :], axis=0), axis=0)
    means = np.delete(means, 0, 0)

    x = (means[0] - mean_values)
    sb = np.append(sb, 5 * np.dot(x.T, x), axis=0)
    sb = np.delete(sb, 0, 0)

    for i in range(1, 40):
        x = (means[i] - mean_values)
        sb += 5 * np.dot(x.T, x)

    for i in range(200):
        Z = np.append(Z, d[i] - means[math.floor(i / 5)], axis=0)
    Z = np.delete(Z, 0, 0)

    tempZ = Z[0:5, :]
    S = np.append(S, np.dot(tempZ.T, tempZ), axis=0)
    S = np.delete(S, 0, 0)

    for i in range(1, 40):
        tempZ = Z[5 * i:5 + 5 * i, :]
        S += np.dot(tempZ.T, tempZ)

    eig_vals, eig_vecs = np.linalg.eigh(np.dot(np.linalg.inv(S), sb))
    Ur.append(eig_vecs[:, 10303])

    for i in range(1, 39):
        Ur.append(eig_vecs[:, 10303 - i])
    Ur = np.array(Ur)
    project_training = np.dot(d, Ur.T)
    project_test = np.dot(test, Ur.T)
    ldaavg = 0
    lda_scores = []
    K = [1, 3, 5, 7]
    for z in range(1, 8, 2):
        model = KNeighborsClassifier(n_neighbors=z, weights='distance')
        model.fit(project_training, labels)
        score = model.score(project_test, labels)
        lda_scores.append(score)
        ldaavg += score
        print("K= ", z)
        print("Accuracy:", score)
    ldaavg = ldaavg/4
    plt.figure(1)
    plt.xlabel("Neighbors")
    plt.ylabel("LDA values")
    plt.plot(K, lda_scores, c='r')
    plt.title("LDA performance against K neighbors ")
    plt.show()
    return ldaavg


def compute_training_testing_data(data_matrix, labels):
    training_data = np.zeros(shape=(1, 10304))
    testing_data = np.zeros(shape=(1, 10304))
    training_labels = []
    testing_labels = []

    for i in range(data_matrix.shape[0]):
        if i % 2 == 0:
            testing_data = np.append(testing_data, data_matrix[i], axis=0)
            testing_labels.append(labels[i])
        else:
            training_data = np.append(training_data, data_matrix[i], axis=0)
            training_labels.append(labels[i])

    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)

    training_data = np.delete(training_data, 0, 0)
    testing_data = np.delete(testing_data, 0, 0)

    return training_labels, testing_labels, training_data, testing_data

def compute_LDA_faces(training_labels, testing_labels, training_data, testing_data):

    Ur = []
    tempZ = np.zeros((5, 10304))
    S = np.zeros((1, 10304))
    means = np.zeros((1, 10304))
    Z = np.zeros((1, 10304))
    for i in range(2):
        means = np.append(means, np.mean(training_data[100 * i:100 + 100 * i, :], axis=0), axis=0)
    means = np.delete(means, 0, 0)

    x = (means[0] - means[1])
    B = np.dot(x.T, x)

    for i in range(200):
        Z = np.append(Z, training_data[i] - means[math.floor(i / 100)], axis=0)
    Z = np.delete(Z, 0, 0)

    tempZ = Z[0:100, :]
    S = np.append(S, np.dot(tempZ.T, tempZ), axis=0)
    S = np.delete(S, 0, 0)

    tempZ = Z[100:200, :]
    S += np.dot(tempZ.T, tempZ)

    eig_vals, eig_vecs = np.linalg.eigh(np.dot(np.linalg.inv(S), B))

    Ur.append(eig_vecs[:, 10303])
    for i in range(1, 3):
        Ur.append(eig_vecs[:, 10303 - i])

    Ur = np.array(Ur)
    project_training = np.dot(training_data, Ur.T)
    project_test = np.dot(testing_data, Ur.T)

    model = KNeighborsClassifier(n_neighbors=1, weights='distance')
    model.fit(project_training, training_labels)
    score = model.score(project_test, testing_labels)
    print(score)

def compute_PCAS(f, training_data, testing_data, training_labels, testing_labels):

    training_eigen_values, training_eigen_vectors = comp_eigens(training_data)
    training_eigen_values = np.flip(training_eigen_values)
    training_eigen_vectors = np.flip(training_eigen_vectors)
    avg = []
    alphas = [0.8, 0.85, 0.9, 0.95]
    if f == 1:
        plt.figure(2)
        plt.xlabel("Neighbors")
        plt.ylabel("PCA values")
    k = [1, 3, 5, 7]
    for i in range(len(alphas)):
        pca_scores = []
        s = 0
        P = PCA(alphas[i], training_eigen_values, training_eigen_vectors)
        projected_training_set = np.dot(training_data, P)
        projected_testing_set = np.dot(testing_data, P)
        for j in range(1, 8, 2):
            print("K=", j)
            knn = KNeighborsClassifier(n_neighbors=j, weights='distance')

            knn.fit(projected_training_set, training_labels)

            print("Model prediction: {}".format(knn.predict(projected_testing_set)))
            score = knn.score(projected_testing_set, testing_labels)
            pca_scores.append(score)
            print("Alpha:{} \nAccuracy:{}".format(alphas[i], score))
            if f == 1:
                s += score
        if f == 1:
            plt.plot(k, pca_scores, label='alpha:{}'.format(alphas[i]))
            avg.append(s/4)
    if f == 1:
        plt.title("PCA performance against K neighbors ")
        plt.legend()
        plt.show()
    if f == 1:
        return np.max(np.array(avg))
    else:
        plot_accuracy(projected_training_set, projected_testing_set, training_labels, testing_labels)


def plot_accuracy(projected_training_set, projected_testing_set, training_labels, testing_labels):
    ranges = [25, 50, 75, 100]
    pca_scores = []
    plt.figure(3)
    plt.xlabel("Number of nonfaces taken")
    plt.ylabel("Accuracy")

    for i in range(125, 201, 25):
        knn = KNeighborsClassifier(n_neighbors=1, weights='distance')
        knn.fit(projected_training_set[:i, :], training_labels[:i])

        print("Model prediction: {}".format(knn.predict(projected_testing_set)))
        score = knn.score(projected_testing_set, testing_labels)
        pca_scores.append(score)
        print("\nAccuracy:{}".format(score))
    plt.plot(ranges, pca_scores, label='ranges:{}'.format(ranges))
    plt.legend()
    plt.show()



if __name__ == '__main__':
    scan_folder('Pics', '.pgm')
    data = np.delete(data, 0, 0)
    labels = []
    count = 1
    for i in range(1, 401):
        labels.append(count)
        if i % 10 == 0:
            count += 1

    labels = np.array(labels)
    print("Computing PCA for Faces only: ")
    training_labels, testing_labels, training_data, testing_data = compute_training_testing_data(data, labels)
    PCA_AVG = compute_PCAS(1, training_data, testing_data, training_labels, testing_labels)

    LDA_AVG = LDA(training_data, testing_data, training_labels)
    print("PCA avg:", PCA_AVG)
    print("LDA avg: ", LDA_AVG)
    if LDA_AVG > PCA_AVG:
        print("Since LDA average score is higher than the PCA average score then we prefer to take LDA")

    else:
        print("Since PCA average score is higher than the LDA average score then we prefer to take PCA")



    # 7- Compare vs Non-face images

    faces_matrix = data.copy()[0:200, :]
    data = np.zeros(shape=(1, 10304))
    scan_folder('nonfaces', '.pgm')
    data = np.delete(data, 0, 0)
    faces_matrix = np.concatenate([faces_matrix, data], axis=0)
    faces_non_faces_labels = []
    for i in range(faces_matrix.shape[0]):
        if i < 200:
            faces_non_faces_labels.append(1)
        else:
            faces_non_faces_labels.append(0)
    faces_non_faces_labels = np.array(faces_non_faces_labels)
    print("Computing PCA for faces and non faces only: ")
    training_labels, testing_labels, training_data, testing_data = compute_training_testing_data(faces_matrix, faces_non_faces_labels)
    PCA_AVG = compute_PCAS(0, training_data, testing_data, training_labels, testing_labels)
    compute_LDA_faces(training_labels, testing_labels, training_data, testing_data)

