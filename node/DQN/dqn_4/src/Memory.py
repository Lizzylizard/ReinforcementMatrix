#!/usr/bin/env python
import numpy as np
from numpy import random


class Memory():

  def __init__(self, memory_size):
    self.memory_size = memory_size
    self.experiences = [0.0] * self.memory_size
    self.count = 0

  def make_tuple(self, state, last_state, action, reward):
    tuple = {"state": state, "last_state": last_state,  "action":
      action, "reward": reward}
    return tuple

  def store_experience(self, state, last_state, action, reward):
    experience_tuple = self.make_tuple(state, last_state, action,
                                       reward)
    # cyclic storing until experiences are full for the first time
    if(self.count <= len(self.experiences)):
      i = self.count % self.memory_size
      self.experiences[i] = experience_tuple
    # random storing
    else:
      i = np.random.randint(0, self.memory_size)
      self.experiences[i] = experience_tuple
    self.count += 1

  def get_random_experience(self, batch_size):
    experience = []
    if(self.count > batch_size and self.count >= len(self.experiences)):
      rand = random.randint(low=0, high=len(self.experiences),
                                size=batch_size)
    elif(self.count > batch_size and self.count < len(
      self.experiences)):
      rand = random.randint(low=0, high=self.count,
                                size=batch_size)
    else:
      rand = random.randint(low=0, high=self.count,
                            size = self.count)
    for i in range(len(rand)):
      experience.append(self.experiences[rand[i]])

    # for easier use of tensorflow ALWAYS return number of
    # experiences = batch size
    # if not enough experiences there, just append the last one
    # until full
    # last_one = experience[-1]
    '''
    if(len(experience) > 1):
      last_one = experience[len(experience)-1]
      while(len(experience) < batch_size):
        experience.append(last_one)
    '''

    # print("Batch size = " + str(batch_size))
    # print("Experience array = \n" + str(experience))
    return experience