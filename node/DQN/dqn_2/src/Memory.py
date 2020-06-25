#!/usr/bin/env python3
import numpy as np
from numpy import random


class Memory():

  def __init__(self, memory_size):
    self.memory_size = memory_size
    self.experiences = []
    self.count = 0

  def make_tuple(self, state, last_state, action, reward):
    tuple = {"state": state, "last_state": last_state,  "action":
      action, "reward": reward}
    return tuple

  def store_experience(self, state, last_state, action, reward):
    experience_tuple = self.make_tuple(state, last_state, action,
                                       reward)
    i = self.count % self.memory_size
    if not(self.count == 0):  # do not store, if initial state
      if((self.count-1) >= self.memory_size):
        self.experiences[i] = experience_tuple
      else:
        self.experiences.append(experience_tuple)
    self.count += 1
    # print("All experiences = \n" + str(self.experiences))

  def get_random_experience(self, batch_size):
    experience = []
    if(len(self.experiences) > batch_size):
      rand = random.randint(low=0, high=len(self.experiences),
                                size=batch_size)
    else:
      rand = random.randint(low=0, high=len(self.experiences),
                            size = len(self.experiences))
    # print("Rand = " + str(rand))
    for i in range(len(rand)):
      # print("rand[i] = " + str(rand[i]))
      experience.append(self.experiences[rand[i]])
    # print("Experience array =\n" + str(experience))
    return experience