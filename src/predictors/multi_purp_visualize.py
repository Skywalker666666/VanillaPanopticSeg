import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

outdir = 'outdir'
i = 0
outname = 'predict_mask_' + str(i) + '.png'
mmask = mpimg.imread(os.path.join(outdir, outname))
plt.imshow(mmask)
plt.show()




