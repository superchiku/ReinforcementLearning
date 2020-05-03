
import time
notebookstart= time.time()
import tensorflow as tf 
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os , sys
from tensorflow.keras.models import load_model
from os import walk
import cv2
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
#from skimage.util.montage import montage2d as montage
#montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datetime import datetime
import operator
from sklearn.preprocessing import MinMaxScaler
from numpy import savetxt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re  #regular expression
from bs4 import BeautifulSoup
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from glob import glob
import gc
#reinforcement learning imports
import gym
import random
from collections import deque
from gym import wrappers
from IPython import display
#import the mobilenetv2 feature extractor
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


file1 = open("Logfile.txt", "a")
def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    clipping_val = 0.2
    critic_discount = 0.5
    #entropy_beta = 0.01
    entropy_beta = 0.001
    def loss(y_true, y_pred):

        newpolicy_probs = y_pred
        #newpolicy_probs = K.sum(y_true * y_pred, axis=1)

        #oldpolicy_probs = K.sum(y_true * oldpolicy_probs, axis=-1)
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss

class PPO:
    def __init__(self, env):
        self.env = env
        self.stack_depth = 4
        self.seq_memory = deque(maxlen=self.stack_depth)
        #self.clipping_val = 0.2
        #self.critic_discount = 0.5
        #self.entropy_beta = 0.001
        self.gamma = 0.99
        self.lmbda = 0.95
        self.model_actor = ''
        self.model_critic = ''

    def to_grayscale(self,img):
        return np.mean(img, axis=2).astype(np.uint8)

    def downsample(self,img):
        return img[::2, ::2]

    def preprocess(self,img):
        #cv2.imshow('image',img)
        width = img.shape[1]
        height = img.shape[0]
        dim = (abs(width/2),abs(height/2))
        resized = cv2.resize(img,(80,105) ) #interpolation = cv2.INTER_AREA)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        resized = resized.reshape(resized.shape+(1,))
        #cv2.imshow('image',np.array(np.squeeze(resized)))
        #resized = self.to_grayscale(resized)
        return(resized)
        #return self.to_grayscale(self.downsample(img)).reshape(1,105,80,1)

    def get_model_actor_image(self,input_dims, output_dims):

        state_input = Input(shape=input_dims)
        oldpolicy_probs = Input(shape=(1, output_dims,))
        advantages = Input(shape=(1, 1,))
        rewards = Input(shape=(1, 1,))
        values = Input(shape=(1, 1,))

        feature_extractor = MobileNetV2(include_top=False, weights='imagenet')

        for layer in feature_extractor.layers:
            layer.trainable = False

        # Classification block
        x = Flatten(name='flatten')(feature_extractor(state_input))
        x = Dense(1024, activation='relu', name='fc1')(x)
        out_actions = Dense(output_dims, activation='softmax', name='predictions')(x)

        model = tensorflow.keras.Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                      outputs=[out_actions])
        optimizer = tensorflow.keras.optimizers.Adam(lr=1e-4)
        model.compile(optimizer, loss=[ppo_loss(
            oldpolicy_probs=oldpolicy_probs,
            advantages=advantages,
            rewards=rewards,
            values=values)], experimental_run_tf_function=False)
        print(model.summary())
        return model

    def get_model_critic_image(self,input_dims):
        state_input = Input(shape=input_dims)

        feature_extractor = MobileNetV2(include_top=False, weights='imagenet')

        for layer in feature_extractor.layers:
            layer.trainable = False

        # Classification block
        x = Flatten(name='flatten')(feature_extractor(state_input))
        x = Dense(1024, activation='relu', name='fc1')(x)
        out_actions = Dense(1, activation='tanh')(x)
        model = tensorflow.keras.Model(inputs=[state_input], outputs=[out_actions])
        optimizer = tensorflow.keras.optimizers.Adam(lr=1e-4)
        model.compile(optimizer, loss='mse')
        print(model.summary())
        return model

    def test_reward(self):
        self.seq_memory.clear()
        cur_state = self.env.reset()
        n_actions = self.env.action_space.n
        dummy_n = np.zeros((1, 1, n_actions))
        dummy_1 = np.zeros((1, 1, 1))
        done = False
        total_reward = 0
        print('testing...')
        file1.writelines('testing...\n')
        limit = 0
        #cur_state = self.preprocess(cur_state)
        # frame = frame.reshape(1, frame.shape[0], frame.shape[1])
        # cur_state = np.repeat(frame, dqn_agent.stack_depth, axis=0)
        self.seq_memory.append(cur_state)
        self.seq_memory.append(cur_state)
        self.seq_memory.append(cur_state)
        self.seq_memory.append(cur_state)
        #cur_state = np.asarray(ppo_agent.seq_memory)
        cur_state = self.seq_memory[3] - self.seq_memory[0]
        #cur_state = cur_state.reshape((1,) + cur_state.shape)
        for itr in range(30):
            new_state = ''
            reward = ''
            done = False
            action = 0
            new_state, reward, done, _ = self.env.step(action)
            #print("initializing -> Action: {}".format(action))
            #new_state = self.preprocess(new_state)
            self.seq_memory.append(new_state)
            #new_state = np.asarray(self.seq_memory)
            #cur_state = new_state
            cur_state = self.seq_memory[3] - self.seq_memory[0]
            #cur_state = cur_state.reshape((1,) + cur_state.shape)
        while not done:
            #state_input = K.expand_dims(state, 0)
            action_probs = self.model_actor.predict([cur_state.reshape((1,) + cur_state.shape), dummy_n, dummy_1, dummy_1, dummy_1])
            action = np.argmax(action_probs)
            new_state, reward, done, _ = self.env.step(action)
            #new_state = self.preprocess(new_state)
            self.seq_memory.append(new_state)
            #new_state = np.asarray(self.seq_memory)
            #cur_state = new_state
            cur_state = self.seq_memory[3] - self.seq_memory[0]
            #cur_state = cur_state.reshape((1,) + cur_state.shape)
            total_reward += reward
            limit += 1
            if limit > 2000:
                break
        return total_reward

    def one_hot_encoding(self,probs):
        one_hot = np.zeros_like(probs)
        one_hot[:, np.argmax(probs, axis=1)] = 1
        return one_hot

    def get_advantages(self,values, masks, rewards):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)



def main():
    #env = gym.make("MsPacmanDeterministic-v4")
    env = gym.make("SpaceInvadersDeterministic-v4")
    env._max_episode_steps = 5000
    ppo_steps = 158
    target_reached = False
    best_reward = 0
    iters = 0
    Trials = 500
    state = env.reset()
    state_dims = env.observation_space.shape
    n_actions = env.action_space.n
    ppo_agent = PPO(env=env)
    dummy_n = np.zeros((1, 1, n_actions))
    dummy_1 = np.zeros((1, 1, 1))
    tensor_board = TensorBoard(log_dir='logs')
    file1.writelines('\n************* TIME START:'+str(time.time())+'*************\n')
    ppo_agent.model_actor = ppo_agent.get_model_actor_image(input_dims=(210,160,3), output_dims=n_actions)
    ppo_agent.model_critic = ppo_agent.get_model_critic_image(input_dims=(210,160,3))
    #model_actor = ppo_get_model_actor_image(input_dims=state_dims, output_dims=n_actions)
    #model_critic = get_model_critic_image(input_dims=state_dims)

    while not target_reached and iters < Trials:

        states = []
        actions = []
        values = []
        masks = []
        rewards = []
        actions_probs = []
        actions_onehot = []
        state_input = None
        # create initial frame stack
        ppo_agent.seq_memory.clear()
        cur_state = env.reset()
        #cur_state = ppo_agent.preprocess(cur_state)
        # frame = frame.reshape(1, frame.shape[0], frame.shape[1])
        # cur_state = np.repeat(frame, dqn_agent.stack_depth, axis=0)
        ppo_agent.seq_memory.append(cur_state)
        ppo_agent.seq_memory.append(cur_state)
        ppo_agent.seq_memory.append(cur_state)
        ppo_agent.seq_memory.append(cur_state)
        #cur_state = np.asarray(ppo_agent.seq_memory)
        cur_state = ppo_agent.seq_memory[3]-ppo_agent.seq_memory[0]
        #cur_state = cur_state.reshape((1,)+cur_state.shape)
        # dqn_agent.seq_memory.extend(cur_state)

        #do nothing for 1st 30 steps as it takes time to initialize
        for itr in range(30):
            action = 0
            new_state, reward, done, _ = env.step(action)
            print("initializing -> Action: {}".format(action))
            file1.writelines("initializing -> Action: {}\n".format(action))
            #new_state = ppo_agent.preprocess(new_state)
            ppo_agent.seq_memory.append(new_state)
            #new_state = np.asarray(ppo_agent.seq_memory)
            #cur_state = new_state
            cur_state = ppo_agent.seq_memory[3] - ppo_agent.seq_memory[0]
            #cur_state = cur_state.reshape((1,) + cur_state.shape)
        for itr in range(ppo_steps-30):
            #state_input = np.asarray(dqn_agent.seq_memory)#K.expand_dims(state, 0)
            action_dist = ppo_agent.model_actor.predict([cur_state.reshape((1,) + cur_state.shape), dummy_n, dummy_1, dummy_1, dummy_1])
            q_value = ppo_agent.model_critic.predict([cur_state.reshape((1,) + cur_state.shape)])
            action = np.random.choice(n_actions, p=action_dist[0, :])
            action_onehot = np.zeros(n_actions)
            action_onehot[action] = 1

            #observation, reward, done, info = env.step(action)
            new_state, reward, done, info = env.step(action)
            #new_state = ppo_agent.preprocess(new_state)
            ppo_agent.seq_memory.append(new_state)
            #new_state = np.asarray(ppo_agent.seq_memory)
            print(
                'Trial: ' + str(iters) + ' episode: ' + str(itr+30) + ', action dist= '+str(action_dist)+', action dist sum= '+str(sum(sum(action_dist)))+', action=' + str(action) + ', reward=' + str(reward) + ', q val=' + str(q_value))
            file1.writelines('Trial: ' + str(iters) + ' episode: ' + str(itr+30) + ', action dist= '+str(action_dist)+', action dist sum= '+str(sum(sum(action_dist)))+', action=' + str(action) + ', reward=' + str(reward) + ', q val=' + str(q_value)+'\n')
            mask = not done
            #mask = done
            states.append(cur_state)
            actions.append(action)
            actions_onehot.append(action_onehot)
            values.append(q_value)
            masks.append(mask)
            rewards.append(reward)
            actions_probs.append(action_dist)
            #print('action distribution:',action_dist.shape)
            #cur_state = new_state
            cur_state = ppo_agent.seq_memory[3] - ppo_agent.seq_memory[0]
            #cur_state = cur_state.reshape((1,) + cur_state.shape)
            if done:
                env.reset()

        q_value = ppo_agent.model_critic.predict(cur_state.reshape((1,) + cur_state.shape))
        values.append(q_value)
        returns, advantages = ppo_agent.get_advantages(values, masks, rewards)
        actions_onehot = np.array(actions_onehot)
        values = np.array(values)
        rewards = np.array(rewards)
        actions_probs = np.array(actions_probs)
        states=np.array(states)
        returns = np.array(returns)

        print('State shape:' + str(states.shape) + ' actions probs:'+str(actions_probs.shape)+ ' advantages:'+str(advantages.shape)+
              ' rewards:'+str(np.reshape(rewards, newshape=(-1, 1, 1)).shape)+ ' values:'+str(values[:-1].shape )+ ' actions one hot: '
              + str((np.reshape(actions_onehot, newshape=(-1, n_actions))).shape)+ ' returns: '
              + str(np.reshape(returns, newshape=(-1, 1)).shape))
        file1.writelines('State shape:' + str(states.shape) + ' actions probs:'+str(actions_probs.shape)+ ' advantages:'+str(advantages.shape)+
              ' rewards:'+str(np.reshape(rewards, newshape=(-1, 1, 1)).shape)+ ' values:'+str(values[:-1].shape )+ ' actions one hot: '
              + str((np.reshape(actions_onehot, newshape=(-1, n_actions))).shape)+ ' returns: '
              + str(np.reshape(returns, newshape=(-1, 1)).shape)+'\n')
        print("training actor")
        file1.writelines("training actor\n")
        actor_loss = ppo_agent.model_actor.fit(
            [states, actions_probs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
            [(np.reshape(actions_onehot, newshape=(-1, n_actions)))], verbose=True, shuffle=True, epochs=8,
            callbacks=[tensor_board])
        print("training critic")
        file1.writelines("training critic\n")
        critic_loss = ppo_agent.model_critic.fit([states], [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
                                                  verbose=True, callbacks=[tensor_board])

        avg_reward = np.mean([ppo_agent.test_reward() for _ in range(5)])
        print('total test reward=' + str(avg_reward))
        file1.writelines('total test reward=' + str(avg_reward)+'\n')
        if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            file1.writelines('best reward=' + str(avg_reward)+'\n')
            ppo_agent.model_actor.save('SavedModels\\model_actor_{}_{}.hdf5'.format(iters, avg_reward))
            ppo_agent.model_critic.save('SavedModels\\model_critic_{}_{}.hdf5'.format(iters, avg_reward))
            best_reward = avg_reward
        if best_reward > 850 or iters > Trials:
            target_reached = True
        iters += 1
        env.reset()

    env.close()



if __name__ == "__main__":
    #ani = animation.FuncAnimation(fig, animate, interval=1000)
    #plt.show()
    main()



file1.writelines("Notebook Runtime: %0.2f Minutes\n"%((time.time() - notebookstart)/60))
file1.writelines('******************END OF LOG******************\n')
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
file1.close()


