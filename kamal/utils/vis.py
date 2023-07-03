import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import pandas as pd
import os
import io

def tsne_features( real_features, real_labels, fake_features, fake_labels, output_file ):
    fig = plt.figure(figsize=(10, 10))
    features = np.concatenate( [real_features, fake_features ], axis=0 )
    labels = np.concatenate( [ real_labels, fake_labels ], axis=0 ).reshape(-1, 1)
    tsne = TSNE( n_components=2, perplexity=10 ).fit_transform( features )
    df = np.concatenate( [tsne, labels], axis=1 )
    df = pd.DataFrame(df, columns=["x", "y", "label"])
    style = [ 'real' for _ in range(len(real_features)) ] + [ 'fake' for _ in range(len(fake_features))]
    sns.scatterplot( x="x", y="y", data=df, hue="label", palette=sns.color_palette("dark", 10), s=50, style=style)
    if output_file is not None:
        dirname = os.path.dirname( output_file )
        if dirname!='':
            os.makedirs( dirname, exist_ok=True )
        plt.savefig( output_file ) 
    else:
        fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img


