
def GetLog(nums):
    Log = []
    for i in range(0, nums):
        Log.append([])
    return Log  #[[],[],]

def GetFileName(basic, nums):
    Names = []
    for i in range(0, nums):
        Names.append(basic + str(i) + '.csv')
    return Names
