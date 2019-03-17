#Classification Using Tensornets
import os
import cv2
import glob
import random
import numpy as np
from skimage import io
from skimage import util
from skimage import transform
from tqdm import tqdm
import tensornets as nets
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


path=r'C:\Users\Anu\Downloads\Compressed\v2-plant-seedlings-dataset'

path_content=os.listdir(path)
images=[]
labels=[]
epochs=20
batch_size=5

#Load Images,Labels
for sample_cls,sample in enumerate(path_content):
    files=glob.glob(os.path.join(path,sample,'*.png'))
    print('Loading Sample.. %d/12 '%(sample_cls+1))
    for file in tqdm(files):
        tmp=[]
        img=io.imread(file)
        if img.shape[2]==4: #Since Some images have 4 channels
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        img=cv2.resize(img,(175,175),cv2.INTER_AREA)
        #Agumentation
        degree=random.uniform(-25,25)
        img_1=transform.rotate(img,degree) #Rotate
        img_2=util.random_noise(img)
        img_3=img[:,::-1]
        images.append(img)
        images.append(img_1)
        images.append(img_2)
        images.append(img_3)
        for lbl in range(4):
            tmp.append(sample_cls)
        labels.extend(tmp)
        
#One-hot Encode labels
print('One_hot Encoding Labels')
lbls=np.zeros((len(labels),12))
for index,label in enumerate(labels):
    lbls[index][label]=1.0
    
#Split Data
print('Data Splitting')
train_images,val_images,train_labels,val_labels=train_test_split(images,lbls,test_size=0.2)


#Model Creation
inputs=tf.placeholder(tf.float32,[None,175,175,3],name='input_images')
true_labels=tf.placeholder(tf.float32,[None,12],name='True_labels')
preds=nets.ResNet50v2(inputs,is_training=True,classes=12)

#Loss and Optimizer
loss=tf.reduce_mean(tf.losses.softmax_cross_entropy(true_labels,preds))
train_op=tf.train.AdamOptimizer(0.0001).minimize(loss)

acc=tf.metrics.accuracy(labels=tf.argmax(true_labels,1),predictions=tf.argmax(preds,1))
saver=tf.train.Saver()

with tf.Session() as sess:
    train_batches=len(train_images)//batch_size
    val_batches=len(val_images)//batch_size
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(epochs):
        print('======================================')
        print('Epoch: ',(epoch+1))
        print('In Training..')
        for batch in tqdm(range(train_batches+1)):
            if batch==train_batches:
                temp_batch=train_images[batch*batch_size:]
                train_image_batch=np.reshape(temp_batch,(len(temp_batch),175,175,3))
                train_label_batch=train_labels[batch*batch_size:]
            else:
                temp_batch=train_images[batch*batch_size:(batch+1)*batch_size]
                train_image_batch=np.reshape(temp_batch,(len(temp_batch),175,175,3))
                train_label_batch=train_labels[batch*batch_size:(batch+1)*batch_size]
            
            train_acc,train_loss,_=sess.run([acc,loss,train_op],feed_dict={inputs:train_image_batch,
                                           true_labels:train_label_batch})
    
        
        #Validation
        print('Running Validation..')
        for batch in tqdm(range(val_batches+1)):
            if batch==val_batches:
                temp_batch=val_images[batch*batch_size:]
                val_image_batch=np.reshape(temp_batch,(len(temp_batch),175,175,3))
                val_label_batch=val_labels[batch*batch_size:]
            else:
                temp_batch=val_images[batch*batch_size:(batch+1)*batch_size]
                val_image_batch=np.reshape(temp_batch,(len(temp_batch),175,175,3))
                val_label_batch=val_labels[batch*batch_size:(batch+1)*batch_size]
            val_acc,val_loss=sess.run([acc,loss],feed_dict={inputs:val_image_batch,
                                      true_labels:val_label_batch})
        print('Train Loss: ',train_loss)
        print('Train Accuracy: ',train_acc)
        print('Validation Loss: ',val_loss)
        print('Validation Accuracy: ',val_acc)
    print('-----Training Finished\n----')
    saver.save(sess, os.path.join(os.getcwd(),"CNN_SC.ckpt"))
                
    
    
       
    

