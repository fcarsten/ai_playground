import sys
import ctypes

ctypes.WinDLL('./lib/cudnn64_6.dll')
for p in sys.path:
    print(p)

import tic_tac_toe.toe4_refactored

if __name__ == '__main__':
    tic_tac_toe.toe4_refactored.main()