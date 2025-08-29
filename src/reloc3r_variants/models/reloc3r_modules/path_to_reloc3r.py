import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
RELOC3R_PATH = path.normpath(path.join(HERE_PATH, "../../../../third_party/reloc3r"))

sys.path.insert(0, RELOC3R_PATH)