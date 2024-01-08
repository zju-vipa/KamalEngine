import os
import sys
import json

root = "./Stanford_Dogs_of_teacher2"
target = "./Stanford_Dogs_of_teacher1"
assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
classlist = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
classlist.sort()
for cla in classlist:
    if os.path.exists(os.path.join(target, cla)) is False:
        os.makedirs(os.path.join(target, cla))