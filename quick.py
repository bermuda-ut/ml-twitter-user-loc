import os
import sys

a = sys.argv[1]
os.system("python3 main.py train build " + a)
os.system("python3 main.py dev build " + a)
os.system("python3 main.py train learn " + a + " 0")
os.system("python3 main.py train feng " + a)
os.system("python3 main.py dev score " + a + " 0")
