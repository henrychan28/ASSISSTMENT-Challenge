import os
import random
import time
import tensorflow as tf
import numpy as np
import pickle
import pandas
from numpy.random import permutation as perm
from sklearn.metrics import roc_curve, auc

####
# missed: mini-batch, N-fold cross validation
# save the model: the structure of neural network, the weight matrix, the bias
# run on the gpu
# other rnn structure: bi-direction rnn
####


###########################  Data Access and Pre-process ########################### 
def read_data(filename):
    # num_prob shall be the info from the dataset.
    # num_steps_max shall not be specified.
    records = []
    vector_gen = []
    num_steps_max = 0
    num_probs = 0
    
    with open(filename, 'r') as f:
        num_steps, seq_probs, seq_tags = None, None, None
        for i, row in enumerate(f):
            try:
                row_0 = row
                row = list(map(int, row.strip().split(",")))
                if i % 4 == 0:
                    ITEST_id = row[0]
                elif i % 4 == 1:
                    num_steps = row[0]
                elif i % 4 == 2:
                    seq_probs = row
                elif i % 4 == 3:
                    seq_tags = row
                    if (num_steps >= 3) and num_steps and seq_probs and seq_tags:
                        num_steps_max = max(num_steps_max, num_steps)
                        num_probs = max([num_probs] + seq_probs)
                        records += [(num_steps, seq_probs, seq_tags)]
                        vector_gen += [(ITEST_id, num_steps, seq_probs, seq_tags)]

            except:
                if i % 4 == 0:
                    ITEST_id = None
                elif i % 4 == 1:
                    num_steps = None
                elif i % 4 == 2:
                    seq_probs = None
                elif i % 4 == 3:
                    seq_tags = None                    
                print("- broken line in {} : {}".format(i, row_0))                
    return vector_gen, records, num_steps_max, num_probs+1

DATA_DIR = './data'
train_file = os.path.join(DATA_DIR, './train.csv')
test_file = os.path.join(DATA_DIR, './test.csv')
vector_gen_train, records_train, num_steps_max_train, num_probs_train = read_data(train_file)
vector_gen_test, records_test, num_steps_max_test, num_probs_test = read_data(test_file)
with open("./data/assisstment_skill.pkl", "rb") as fp:
    skill_list = pickle.load(fp)
# num_steps = max(num_steps_max_train, num_steps_max_test)
num_probs = max(num_probs_train, num_probs_test)
# num_steps_max_test: 1062
# num_probs_test: 124


def parse(dtype=-1, cv=None, sample=False):
    if sample == False:
        if dtype == -1:
            #print("parse mode: records_train")
            return list(range(len(records_train)))
        if dtype == 0:
            #print("parse mode: records_test")
            return list(range(len(records_test)))
        if dtype == 1:
            return None



def preprocess(idx, dtype=-1):
    # one_hot for both x and y
    if dtype == -1:  # train
        #print("preprocess mode: records_train")
        return records_train[idx]
    elif dtype == 0: # valid
        #print("preprocess mode: records_test")
        return records_test[idx]
    else:
        return



def batch(idxs, dtype=-1, cv=None, sample=False):
    #print("batch mode: {0}".format(dtype))
    if dtype == 1:
        return None, None
    
    s_seq = []
    x_seq = []
    y_seq = []
    for idx in idxs:
        s, x_inp, y_inp = preprocess(idx, dtype)
        s_seq += [s]
        x_seq += [x_inp]
        y_seq += [y_inp]

        

#padding skill_sequences with -1, padding ans_sequences with 0



    num_steps = max(s_seq)
    for i in range(len(x_seq)):
        x_seq[i] = x_seq[i][0:num_steps] + [-1] * (num_steps-len(x_seq[i]))  if len(x_seq[i]) < num_steps else x_seq[i][0:num_steps]
        y_seq[i] = y_seq[i][0:num_steps] + [0] * (num_steps-len(y_seq[i])) if len(y_seq[i]) < num_steps else y_seq[i][0:num_steps]
    
    x_feed = {"X_ph": np.array(x_seq)}
    y_feed = {"Y_ph": np.array(y_seq)}
    
    return x_feed, y_feed


###########################  START RNN ########################### 
# X_ph (seq_probs)   :
# Y_ph (seq_tags)    :
#                    : batch_size x num_steps

def seq_onehot(seq_probs, seq_tags, num_steps, num_probs):
    seq_probs_ = tf.one_hot(seq_probs, depth=num_probs)
    seq_probs_flat = tf.reshape(seq_probs_, [-1, num_probs])
    
    # element-wise multiplication between Matrix and Vector
    # the i-th column of Matrixelement-wisedly multiply the i-th element in the Vector
    
    seq_tags_ = tf.cast(tf.reshape(seq_tags, [-1]), dtype=tf.float32)
    seq_tags_ = tf.multiply(tf.transpose(seq_probs_flat), seq_tags_)
    seq_tags_ = tf.reshape(tf.transpose(seq_tags_), shape=[-1, num_steps, num_probs])
    return seq_tags_ * 2 - seq_probs_, seq_tags_


'''
return :
[batch_size, num_steps, num_probs], 
[s, a, b] = 1 => student s answer problem a correctly, 
[s, a, b] = 0 => ..... did not answer
[s, a, b] = -1=> ...... incorrect
'''

###################
# Hyperparameter###
###################
batch_size = 32
num_layers = 2
state_size = 250
learning_rate=1e-3
dropout_prob = 0.5
###################

X_ph = tf.placeholder(tf.int32, [None, None])
Y_ph = tf.placeholder(tf.int32, [None, None])
keep_prob_ph = tf.placeholder(tf.float32)

num_steps = tf.shape(X_ph)[1]
#print(num_steps)
X_in, Y_in = seq_onehot(X_ph, Y_ph, num_steps, num_probs)

## build up the network
cells = [tf.contrib.rnn.LSTMCell(num_units=state_size, forget_bias=1.0, state_is_tuple=True) for _ in range(num_layers)]
cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob_ph) for cell in cells]

rnn_outputs_in_list = []
rnn_inputs = X_in
for i, cell in enumerate(cells):
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, time_major=False, scope="rnn-layer-"+str(i), dtype = tf.float32)
    rnn_outputs_in_list += [rnn_outputs]
    rnn_inputs = rnn_outputs

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_probs])
    b = tf.get_variable('b', [num_probs], initializer = tf.constant_initializer(0.0))

Y_out = tf.matmul(tf.reshape(tf.tanh(rnn_outputs), [-1, state_size]), W) + b
Y_out = tf.sigmoid(tf.reshape(Y_out, [-1, num_steps, num_probs]))



###########################  Define Loss  ########################### 
# Y_out: batch_size x num_steps x num_probs
# why split?
_, X_in_next = tf.split(X_in, num_or_size_splits = [1, num_steps-1], axis=1)
Y_out_cur, _ = tf.split(Y_out, num_or_size_splits = [num_steps-1, 1], axis=1)
_, Y_in_next = tf.split(Y_in, num_or_size_splits = [1, num_steps-1], axis=1)

# this code block calculate the loss using tf.gather_nd
idx_selected = tf.where(tf.not_equal(X_in_next, 0))
Y_out_selected = tf.gather_nd(Y_out_cur, idx_selected)
Y_in_selected = tf.gather_nd(Y_in_next, idx_selected)

loss = -Y_in_selected * tf.log(Y_out_selected) - (1-Y_in_selected) * tf.log(1-Y_out_selected)
total_loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)


###########################  Mini-batch  ########################### 
def shuffle(dtype=-1, cv=None, mode=-1, epoch=1):
    data = np.array(parse(dtype, cv))
    #print("shuffle mode: {0}: ",mode)
    if data is None: return
    
    size = len(data)
    
    batch_per_epoch = int(size / batch_size) + bool(size % batch_size) * (mode != -1)
    
    total = epoch * batch_per_epoch
    yield total # total batch number
    
    num_epoch = epoch if mode == -1 else 1
    for i in range(num_epoch):        
        idx_shuffle = perm(np.arange(size)) if mode == -1 else np.arange(size)
        
        for b in range(batch_per_epoch):
            if (b+1) == batch_per_epoch and mode == -1:
                idx_in_list = idx_shuffle[b*batch_size:]
            else:
                idx_in_list = idx_shuffle[(b*batch_size):(b+1)*batch_size]
                
            x_batch, y_batch = batch(data[idx_in_list], dtype, cv)
            
            yield (x_batch, y_batch, i, b, (b+1)==batch_per_epoch)



###########################  Evaluation function  ########################### 
def evaluate(sess, mode=-1):
    """
    auc score
    """
    #print("evaluate mode: {0}: ",mode)
    def auc_score(prob_pred, prob_true):
            fpr, tpr, thres = roc_curve(prob_true, prob_pred, pos_label=1)
            return auc(fpr, tpr)

    batches = shuffle(dtype=mode, cv=None, mode=mode, epoch=1)
    
    y_pred = []
    y_true = []
    for i, packet in enumerate(batches):
        if i == 0:
            total = packet
        else:
            x_batch, y_batch, idx_epoch, idx_batch, end_batch = packet
            y_out, y_in = sess.run((Y_out_selected, Y_in_selected),
                                   feed_dict={ X_ph: x_batch["X_ph"],
                                               Y_ph: y_batch["Y_ph"],
                                               keep_prob_ph: 1.0,
                                           }
                               )
            y_pred += [y_out]
            y_true += [y_in]
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    return auc_score(y_pred, y_true)



###########################  Training process  ########################### 
def optimize(sess, num_epochs):
    batches = shuffle(dtype=-1, cv=None, mode=-1, epoch = num_epochs)
    
    for i, packet in enumerate(batches):
        if i == 0:
            total = packet
            auc_train = evaluate(sess, mode=-1)
            auc_test = evaluate(sess, mode=0)
            print(("[eval] Epoch {0:>4},  train auc {1:.5}, test auc: {2:.5}".format(-1, auc_train, auc_test)))
        else:
            x_batch, y_batch, idx_epoch, idx_batch, end_batch = packet
            sess.run(optimizer,
                     feed_dict={ X_ph: x_batch["X_ph"],
                                 Y_ph: y_batch["Y_ph"],
                                 keep_prob_ph:dropout_prob,
                             }
                     )
            
            if idx_batch % 20 == 0:
                total_loss_eval, = sess.run((total_loss, ),
                                            feed_dict={  X_ph: x_batch["X_ph"],
                                                         Y_ph: y_batch["Y_ph"],
                                                         keep_prob_ph: 1.0,
                                                    }
                                        )
                print(("Epoch {0:>4}, iteration {1:>4}, batch loss value: {2:.5}".format(idx_epoch, idx_batch, total_loss_eval)))
            if end_batch:
                auc_train = evaluate(sess, mode=-1)
                auc_test = evaluate(sess, mode=0)
                print(("[eval] Epoch {0:>4}, train auc {1:.5}, test auc: {2:.5}".format(idx_epoch, auc_train, auc_test)))
                save_path = saver.save(sess, "./saved_model/model")
                print("Model saved in file: %s" % save_path)


def extraction(sess, data):
    #input a variable with form [student_id, number of steps, [#skill], [correct]]
    y_out = []
    for i in range(len(data)):
        x_input = data[i][2]
        y_input = data[i][3]
        x_input = np.array(x_input)
        x_input = np.reshape(x_input, (x_input.shape[0], 1))
        y_input = np.array(y_input)
        y_input = np.reshape(y_input, (y_input.shape[0], 1))
        #print(x_input.shape, y_input.shape)
        #feed the variable for training to extract y_out
        temp = sess.run((Y_out),feed_dict={ X_ph: x_input,
                                                   Y_ph: y_input,
                                                   keep_prob_ph: 1.0})

        y_out.append(temp)
        
    #Batch_state: [(ITEST_id1, state_vector1), (ITEST_id2, state_vector2), ....]
    batch_state = [(data[i][0],y_out[i][data[i][1]-1][0]) for i in range(len(y_out))]
    #print(batch_state[0][1])
    #Collect ITEST_id into a dictionary with key [ITEST_id1, ITEST_id2, ITEST_id3, ...]
    output = {"ITEST_id": [data[i][0] for i in range(len(data))]}
    
    #Collect Skills into output with the format {'SKill 1': [probability for student 1, probability for student 2, ...]}
    output.update(dict([(skill_list[i], [row[1][i] for row in batch_state]) for i in range(len(skill_list))]))
    #print(output)
    
    #Put the whole thing into a pandas dataframe
    output = pandas.DataFrame(output)
    
    #Set ITEST_id as the index of the output
    
    return output



WITH_CONFIG = True
num_epochs = 200



#Restore Model
restore = 0  #0: not restore, 1: restore
saver = tf.train.Saver()
if restore:
    imported_meta = tf.train.import_meta_graph("./saved_model/model.meta")


start_time = time.time()
if WITH_CONFIG:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # specify the GPU to run
    config.gpu_options.visible_device_list = '1'
    with tf.Session(config=config) as sess:
        if restore:
            imported_meta.restore(sess, tf.train.latest_checkpoint("./"))
        sess.run(tf.global_variables_initializer())
        optimize(sess, num_epochs)
else:
    with tf.Session() as sess:
        if restore:
            imported_meta.restore(sess, tf.train.latest_checkpoint("./"))
        sess.run(tf.global_variables_initializer())
        optimize(sess, num_epochs)

end_time = time.time()

print(("program run for: {0}s".format(end_time-start_time)))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out = extraction(sess, vector_gen_train)
    
    
with open("./data/assisstment_student_new_state.pkl", "wb") as fp:
    pickle.dump(out, fp)



