#!/usr/bin/env python3
import numpy as np
from numpy import random


class Memory():

  def __init__(self, memory_size):
    self.memory_size = memory_size
    self.experiences = [None] * self.memory_size
    self.count = 0

  def make_tuple(self, state, last_state, action, reward):
    tuple = {state: state, last_state: last_state, action: action, \
            reward: reward}
    return tuple

  def store_experience(self, state, last_state, action, reward):
    experience_tuple = self.make_tuple(state, last_state, action,
                                       reward)
    i = self.count % self.memory_size
    self.experiences[i] = experience_tuple
    self.count += 1

  def get_random_experience(self, batch_size):
    experience = []
    if(len(self.experiences) >= batch_size):
      rand = random.randint(len(self.experiences), size=batch_size)
    else:
      rand = random.randint(len(self.experiences), len(self.experiences))
    for i in range(rand):
      experience.append(self.experiences[rand[i]])
    return experience