#!/usr/bin/env python3
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
    # print("Tuple = " + str(experience_tuple))
    i = self.count % self.memory_size
    # print("i = " + str(i))
    #if not(self.count == 0):  # do not store, if initial state
    # if((self.count-1) >= self.memory_size):
    self.experiences[i] = experience_tuple
    # else:
      # self.experiences.append(experience_tuple)
    self.count += 1
    # print("All experiences = \n" + str(self.experiences))

  def get_random_experience(self, batch_size):
    experience = []
    print("self.count = " + str(self.count))
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
    # print("Rand = " + str(rand))
    for i in range(len(rand)):
      # print("rand[i] = " + str(rand[i]))
      experience.append(self.experiences[rand[i]])
    print("Batch size = " + str(batch_size))
    print("Experience array = \n" + str(experience))
    return experience