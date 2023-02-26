import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

# population on our greyhound and labrodors

grey_height = 28 + 4 * np.random.randn(greyhounds)
labs_height = 24 + 4 * np.random.randn(labs)
# this code will add or subtract 4 from the given height of greyhounds and labrodors

plt.hist([grey_height, labs_height], stacked=True, color=['r', 'b'])
plt.show()
# in this example we learned that a good feature is a must to correctly 
# classifiy our dataset and avoiding redundency is a must to predict an accurate result
# chooing the right distinigushed feature will decide our outcome of the classifiers
