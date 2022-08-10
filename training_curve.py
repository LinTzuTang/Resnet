import matplotlib.pyplot as plt
from matplotlib import gridspec

def history(t_m):
    df = t_m.history
    fig1 = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 2) 
    ax1 = fig1.add_subplot(gs[0,0])
    ax2 = fig1.add_subplot(gs[0,1])

    ax1.set_title('Train Accuracy',fontsize = '14' )
    ax2.set_title('Train Loss', fontfamily = 'serif', fontsize = '18' )
    ax1.set_xlabel('Epoch', fontfamily = 'serif', fontsize = '13' )
    ax1.set_ylabel('Acc', fontfamily = 'serif', fontsize = '13' )
    ax2.set_xlabel('Epoch', fontfamily = 'serif', fontsize = '13' )
    ax2.set_ylabel('Loss', fontfamily = 'serif', fontsize = '13' )
    ax1.plot(df['accuracy'], label = 'train',linewidth=2)
    ax1.plot(df['val_accuracy'], label = 'validation',linewidth=2)
    ax2.plot(df['loss'], label = 'train',linewidth=2)
    ax2.plot(df['val_loss'], label = 'validation',linewidth=2)
    ax1.legend(['train', 'validation'], loc='upper left')
    ax2.legend(['train', 'validation'], loc='upper left')
    plt.show