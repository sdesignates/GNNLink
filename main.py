from inits import load_data1,preprocess_graph
from Transorfomer import TransorfomerModel
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import time


epochs = 100
batch_size = 1

def Main():
    geneName, feature, logits_train, logits_test, logits_validation, train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation, train_data, validation_data, test_data = load_data1()
    biases = preprocess_graph(interaction)
    # v1 = tf.Variable(5, name='v1')
    # saver = tf.compat.v1.train.Saver([v1])
    model = TransorfomerModel(feature,do_train=False)
    with tf.compat.v1.Session() as sess:
        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        sess.run(init_op)
        # saver.restore(sess, tf.train.latest_checkpoint("savemodel/"))
        train_loss_avg = 0
        train_acc_avg = 0

        for epoch in range(epochs):
            t = time.time()
            ######## train #########
            tr_step = 0
            tr_size = 1
            if tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([model.train_op, model.loss, model.accuracy],
                                                    feed_dict={
                                                        model.encoded_gene: feature,
                                                        model.bias_in:biases,
                                                        model.lbl_in: logits_train,
                                                        model.msk_in: train_mask,
                                                        model.neg_msk: neg_logits_train
                                                    })
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1


            score, _, _ = sess.run([model.logits, model.loss, model.accuracy],
                                                feed_dict={
                                                    model.encoded_gene: feature,
                                                    model.bias_in: biases,
                                                    model.lbl_in: logits_validation,
                                                    model.msk_in: validation_mask,
                                                    model.neg_msk: neg_logits_validation
                                                })
            score = score.reshape((feature.shape[0], feature.shape[0]))
            auc_val,aupr_val = evaluate(validation_data, score)
            print("Epoch: %04d | Training: loss = %.5f, acc = %.5f, auc = %.5f, aupr = %.5f, time = %.5f", train_loss_avg, train_acc_avg, auc_val, aupr_val,
                  time.time() - t)
        print("Finish training.")

        ###########     test      ############
        ts_size = 1
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        print("Start to test")
        while ts_step * batch_size < ts_size:
            out_come, loss_value_ts, acc_ts = sess.run([model.logits, model.loss, model.accuracy],
                                                       feed_dict={
                                                           model.encoded_gene: feature,
                                                           model.bias_in: biases,
                                                           model.lbl_in: logits_test,
                                                           model.msk_in: test_mask,
                                                           model.neg_msk: neg_logits_test})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1
        print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

        out_come = out_come.reshape((feature.shape[0], feature.shape[0]))

        return geneName, out_come, test_data
        sess.close()




def evaluate(rel_test_label, pre_test_label):
    temp_pre = []
    for i in range(rel_test_label.shape[0]):
        l = []
        m = rel_test_label[i, 0]
        n = rel_test_label[i, 1]
        l.append(m)
        l.append(n)
        l.append(pre_test_label[m, n])
        temp_pre.append(l)
    temp_pre = np.asarray(temp_pre)
    prec, rec, thr = precision_recall_curve(rel_test_label[:, 2], temp_pre[:, 2])
    aupr_val = auc(rec, prec)
    aupr_vec.append(aupr_val)
    fpr, tpr, thr = roc_curve(rel_test_label[:, 2], temp_pre[:, 2])
    auc_val = auc(fpr, tpr)

    return auc_val,aupr_val





seed= 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

T = 1
cv_num = 1

for t in range(T):
    aupr_vec = []
    auroc_ver = []
    for i in range(cv_num):
        t1 = time.time()
        geneName, pre_test_label, rel_test_label = Main()
        print(time.time() - t1)
        temp_pre = []
        for i in range(rel_test_label.shape[0]):
            l = []
            m = rel_test_label[i,0]
            n = rel_test_label[i,1]
            l.append(m)
            l.append(n)
            l.append(pre_test_label[m, n])
            temp_pre.append(l)
        temp_pre = np.asarray(temp_pre)
        prec, rec, thr = precision_recall_curve(rel_test_label[:,2], temp_pre[:,2])
        aupr_val = auc(rec, prec)
        aupr_vec.append(aupr_val)
        fpr, tpr, thr = roc_curve(rel_test_label[:,2], temp_pre[:,2])
        auc_val = auc(fpr, tpr)

        print("auc:%.6f, aupr:%.6f" % (auc_val, aupr_val))
        plt.figure
        plt.plot(fpr, tpr)
        plt.show()
        plt.figure
        plt.plot(rec, prec)
        plt.show()



