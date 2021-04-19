from scipy import stats
import numpy as np
val_6=[0.74829,0.71496,0.70236,0.70556,0.73340]
val_10=[0.63569,0.66601,0.67226,0.60303,0.66701]
shapiro_test = stats.shapiro(val_6)
print("6",shapiro_test)

shapiro_test = stats.shapiro(val_10)
print("10",shapiro_test)

#indep t test
# https://koreadatascientist.tistory.com/8
print(stats.ttest_ind(val_6,val_10))