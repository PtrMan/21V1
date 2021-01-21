# returns the classification or None if no classification could be determined
def calcClass(arr):
    sel0 = arr[0] > 0.0
    sel1 = arr[1] > 0.0
    if sel0 and sel1: # is selection valid? if not then we just ignore the result!
        n = 2
        maxSelVal = float("-inf")
        maxSelIdx = None
        for iIdx in range(n):
            if arr[iIdx] > maxSelVal:
                maxSelIdx = iIdx
                maxSelVal = arr[iIdx]
        
        return maxSelIdx
    else:
        return None # no classification was possible
