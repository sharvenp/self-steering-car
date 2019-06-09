
import keras
from keras.models import load_model
from keras.models import Model
from keras import layers
from keras import losses
from keras import optimizers
from keras import backend as keras_back
from keras import utils as np_utils

import os
import numpy as np

import colorama  
from termcolor import colored 

class Agent:

    def __init__(self, architecture, lr, gamma, saving_interval, load_most_recent):

        colorama.init() # This enables colored print statements

        # Constant Params
        # Neural Net
        self.LEARNING_RATE = lr
        self.GAMMA = gamma
        self.SAVE_INTERVAL = saving_interval

        self.current_episode = 0
        
        self.input_dim = architecture[0][0]
        self.output_dim = architecture[-1][0]

        self._create_model(architecture, load_most_recent)
        self._create_training_function()

    def _create_model(self, architecture, load):

        # Create agent model or load the last checkpoint

        if not load:
            self.input = layers.Input(shape=(self.input_dim,))
            net = self.input
            
            for h in architecture[1:len(architecture)-1]:
                net = layers.Dense(h[0])(net)
                net = layers.Activation(h[1])(net)

            net = layers.Dense(self.output_dim)(net)
            net = layers.Activation(architecture[-1][1])(net)
            self.model = Model(inputs=self.input, outputs=net)
        
        else:
            list_of_saved_nets = os.listdir('models/')
            if list_of_saved_nets:
                m = -1
                prefix = 'chkpnt-'
                suffix = '.h5'
                for file_name in list_of_saved_nets:
                    k = file_name.replace(prefix, '')
                    k = k.replace(suffix, '')
                    chkpoint = int(k)
                    if chkpoint > m:
                        m = chkpoint
                
                self.current_episode = m
                fn = prefix+str(m)+suffix
                self._load_model('models/'+fn)
            else:
                print(colored('FATAL: MODELS DIRECTORY EMPTY', 'red')) 
                quit(0)
                
    def _load_model(self, directory):
        # Load a keras agent
        self.model = load_model(directory)
        print(colored('Loaded Model From: ' + directory, 'blue')) 

    def _save_model(self, directory):
        # Save the agent
        self.model.save(directory)
        print(colored('Saved Model To: ' + directory, 'green')) 

    def _create_training_function(self):
        
        # Create a Reinforcement Learning training function
        action_prob_placeholder = self.model.output 
        action_onehot_placeholder = keras_back.placeholder(shape=(None, self.output_dim), name="action_onehot")
        discount_reward_placeholder = keras_back.placeholder(shape=(None,), name="discount_reward")

        action_prob = keras_back.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = keras_back.log(action_prob)

        loss = -1*log_action_prob * discount_reward_placeholder
        loss = keras_back.mean(loss)

        adam = optimizers.Adam(lr=self.LEARNING_RATE)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.training_function = keras_back.function(inputs=[self.model.input, action_onehot_placeholder,
                                                     discount_reward_placeholder], outputs=[],
                                                     updates=updates)

    def get_state_action(self, state_input):

        # Get agent action from a given state
        action_prob = np.squeeze(self.model.predict(np.asarray([state_input])))
        val = np.random.choice(np.arange(self.output_dim), p=action_prob)
        # val = np.argmax(action_prob) # Greedy action
        return val
    
    def _train (self, S, A, R):

        # Train the agent with given states <S>, actions <A> and rewards <R>
        action_onehot = np_utils.to_categorical(A, num_classes=self.output_dim)
        discount_reward = self._compute_discounted_rewards(R)
        self.training_function([S, action_onehot, discount_reward])

    def train_episode(self, s, a, r, game, save=True):

        # Wrapper for _train that also saves
        if s and a and r:
            # Theres a bug where len(r) > len(s) so quick kludge around it
            while len(r) > len(s): 
                r.pop()    
            directory = 'models/'
            states = np.asarray(s)
            actions = np.asarray(a)
            rewards = np.asarray(r)
            self._train(states, actions, rewards)
            if game % self.SAVE_INTERVAL == 0 and game and save:
                self._save_model(directory+"chkpnt-"+str(game)+".h5")

    def _compute_discounted_rewards (self, R):

        # Computes discounted rewards based on R and GAMMA
        discounted_r = np.zeros_like(R, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(R))):
            running_add = running_add * self.GAMMA + R[t]
            discounted_r[t] = running_add
        discounted_r -= (discounted_r.mean() / discounted_r.std())
        return discounted_r

