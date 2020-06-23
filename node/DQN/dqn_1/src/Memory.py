#!/usr/bin/env python3
import numpy as np
from numpy import random


class Memory():
  self.experiences = []
  self.memory_size = None

  def __init__(self, memory_size):
    self.memory_size = memory_size
    self.experiences = np.zeros(shape=[self.memory_size])

  def make_tupel(self, state, last_state, action, reward):
    tupel = {state: state, last_state: last_state, action: action, \
            reward: reward}
    return tupel

  def store_experience(self, state, last_state, action, reward):
    experience_tupel = self.make_tupel(state, last_state, action,
                                       reward)
    if(len(self.experiences) < self.memory_size):
      self.experiences.append(experience_tupel)
    else:
      self.experiences.pop(0)
      self.experiences.append(experience_tupel)

  def get_random_experience(self, batch_size):
    experience = []
    if(len(self.experiences) >= batch_size):
      rand = random.randint(len(self.experiences), size=batch_size)
    else:
      rand = random.randint(len(self.experiences), len(self.experiences))
    for i in range(rand):
      experience.append(self.experiences[rand[i]])
    return experience